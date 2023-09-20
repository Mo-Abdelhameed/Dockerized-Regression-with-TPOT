import numpy as np

from src.preprocessing.preprocess import encode, impute_categorical, impute_numeric


def test_encode(sample_train_data, schema_provider):
    """Tests that the data gets one hot encoded"""
    sample_train_data.drop(
        columns=[schema_provider.id, schema_provider.target], inplace=True
    )
    total_num_categories = 0
    for f in schema_provider.categorical_features:
        total_num_categories += len(
            schema_provider.get_allowed_values_for_categorical_feature(f)
        )
        sample_train_data[f] = sample_train_data[f].astype(str)
    encoded_data = encode(sample_train_data, schema_provider)
    assert encoded_data.shape[1] > sample_train_data.shape[1]


def test_impute_numeric(sample_train_data):
    """Tests numeric imputations"""
    sample_train_data.loc[0:5, "numeric_feature_1"] = np.nan
    assert sample_train_data.isna().any()["numeric_feature_1"]

    sample_train_data, _ = impute_numeric(sample_train_data, column="numeric_feature_1")
    assert not sample_train_data.isna().any()["numeric_feature_1"]


def test_impute_categorical(sample_train_data):
    """Tests categorical imputations"""
    sample_train_data.loc[0:5, "categorical_feature_1"] = np.nan
    assert sample_train_data.isna().any()["categorical_feature_1"]

    sample_train_data, _ = impute_categorical(
        sample_train_data, column="categorical_feature_1"
    )
    assert not sample_train_data.isna().any()["categorical_feature_1"]
