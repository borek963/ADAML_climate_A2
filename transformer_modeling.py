import pandas as pd
import tensorflow as tf
import torch
import numpy as np
from typing import Optional, Iterable
import matplotlib.pyplot as plt
from functools import partial

from datasets import Dataset, DatasetDict

from gluonts.time_feature import (time_features_from_frequency_str, TimeFeature, get_lags_for_frequency)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (AddAgeFeature, AddObservedValuesIndicator, AddTimeFeatures, AsNumpyArray, Chain,
                               ExpectedNumInstanceSampler, InstanceSplitter, RemoveFields, SelectFields, SetField,
                               TestSplitSampler, Transformation, ValidationSplitSampler, VstackFeatures, RenameFields
                               )
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

from transformers import PretrainedConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from evaluate import load

from accelerate import Accelerator
from torch.optim import AdamW


def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch


def create_instance_splitter(
        config: PretrainedConfig,
        mode: str,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
                 or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
                      or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                             + (
                                 [FieldName.FEAT_DYNAMIC_REAL]
                                 if config.num_dynamic_real_features > 0
                                 else []
                             ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_train_dataloader(
        config: PretrainedConfig,
        freq,
        data,
        batch_size: int,
        num_batches_per_epoch: int,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = True,
        **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=True
    )

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_test_dataloader(
        config: PretrainedConfig,
        freq,
        data,
        batch_size: int,
        **kwargs
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


def plot(ts_index):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=validation_dataset[ts_index][FieldName.START],
        periods=len(validation_dataset[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    ax.plot(
        index,
        validation_dataset[ts_index]["target"],
        label="actual",
    )

    plt.plot(
        index,
        np.median(forecasts[ts_index], axis=0),
        label="median",
    )

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("preprocessed_dataset.csv")
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Resample the data by month and get the values for each month
    monthly_data = df.resample('M').agg(lambda x: x.tolist())

    # Calculate the index to split the data into train and validation (80-20 split)
    # split_index = int(31 / 2)
    # split_val = int(31 / 2 + 31 / 4)

    split_val = int(np.where((monthly_data.index == "2015-12-31"))[0])

    # Create 'train' and 'validation' dictionaries
    train_dataset = {
        'start': monthly_data.index[:split_val],
        'target': [month for month in monthly_data[:split_val]['humidity']],
        'item_id': list(range(1, len(monthly_data[:split_val]) + 1)),
        'feat_dynamic_real': len(monthly_data[:split_val]) * [None],
        'feat_static_cat': list(range(1, len(monthly_data[:split_val]) + 1))
    }

    validation_dataset = {
        'start': monthly_data.index[split_val:],
        'target': [month for month in monthly_data[split_val:]['humidity']],
        'item_id': list(range(1, len(monthly_data[split_val:]) + 1)),
        'feat_dynamic_real': len(monthly_data[split_val:]) * [None],
        'feat_static_cat': list(range(1, len(monthly_data[split_val:]) + 1))
    }

    test_dataset = {
        'start': monthly_data.index[split_val:],
        'target': [month for month in monthly_data[split_val:]['humidity']],
        'item_id': list(range(1, len(monthly_data[split_val:]) + 1)),
        'feat_dynamic_real': len(monthly_data[split_val:]) * [None],
        'feat_static_cat': list(range(1, len(monthly_data[split_val:]) + 1))
    }

    # Create Dataset objects
    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)
    validation_dataset = Dataset.from_dict(validation_dataset)

    # Create the DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    # Display or use the dataset_dict as needed
    print(dataset)

    freq = "1D"
    prediction_length = int(31)
    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))

    lags_sequence = get_lags_for_frequency(freq)
    print(lags_sequence)

    time_features = time_features_from_frequency_str(freq)
    print(time_features)

    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        # context length:
        context_length=prediction_length * 2,
        # lags coming from helper given the freq:
        lags_sequence=lags_sequence,
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(time_features) + 1,
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=0,
        # it has 366 possible values:
        cardinality=[len(train_dataset)],
        # the model will learn an embedding of size 2 for each of the 366 possible values:
        embedding_dimension=[2],

        # transformer params:
        encoder_layers=4,
        decoder_layers=4,
        d_model=64,
    )

    model = TimeSeriesTransformerForPrediction(config)
    print(model.model.encoder.layers)

    train_dataloader = create_train_dataloader(
        config=config,
        freq=freq,
        data=train_dataset,
        batch_size=64,
        num_batches_per_epoch=100,
    )

    test_dataloader = create_test_dataloader(
        config=config,
        freq=freq,
        data=test_dataset,
        batch_size=64,
    )

    batch = next(iter(train_dataloader))
    for k, v in batch.items():
        print(k, v.shape, v.type())

    # perform forward pass
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"]
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"]
        if config.num_static_real_features > 0
        else None,
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
        future_observed_mask=batch["future_observed_mask"],
        output_hidden_states=True,
    )

    accelerator = Accelerator()
    device = accelerator.device

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    model.train()
    for epoch in range(200):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Ensure the input tensors have the correct data type
            static_categorical_features = (
                batch["static_categorical_features"]
                .to(device)
                .to(torch.float)  # Convert to float if not already
                if config.num_static_categorical_features > 0
                else None
            )

            static_real_features = (
                batch["static_real_features"]
                .to(device)
                .to(torch.float)  # Convert to float if not already
                if config.num_static_real_features > 0
                else None
            )

            past_time_features = batch["past_time_features"].to(device)
            past_values = batch["past_values"].to(device)
            future_time_features = batch["future_time_features"].to(device)
            future_values = batch["future_values"].to(device)
            past_observed_mask = batch["past_observed_mask"].to(device)
            future_observed_mask = batch["future_observed_mask"].to(device)

            outputs = model(
                static_categorical_features=static_categorical_features,
                static_real_features=static_real_features,
                past_time_features=past_time_features,
                past_values=past_values,
                future_time_features=future_time_features,
                future_values=future_values,
                past_observed_mask=past_observed_mask,
                future_observed_mask=future_observed_mask,
            )
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            if idx % 100 == 0:
                print(loss.item())

    model.eval()
    forecasts = []

    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
    forecasts.append(outputs.sequences.cpu().numpy())
    forecasts = np.vstack(forecasts)

    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, 1)

    plot(0)
