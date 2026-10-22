import datetime as dt
from types import SimpleNamespace

import pytest

pl = pytest.importorskip("polars")

from enefit_prosumer.features import FeatureEngineer


def test_general_features_create_segment_and_cyclical_columns():
    engineer = FeatureEngineer(SimpleNamespace())
    df = pl.DataFrame(
        {
            "county": [0],
            "is_business": [1],
            "product_type": [2],
            "is_consumption": [0],
            "datetime": [dt.datetime(2022, 1, 1, 3)],
            "row_id": [123],
        }
    )

    result = engineer._add_general_features(df)

    assert result["segment"].to_list() == ["0_1_2_0"]
    assert {"sin_dayofyear", "cos_dayofyear", "sin_hour", "cos_hour"}.issubset(result.columns)
