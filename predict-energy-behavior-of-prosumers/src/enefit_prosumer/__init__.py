"""Reusable utilities for the Enefit prosumer forecasting project."""

__all__ = ["DataStorage", "FeatureEngineer", "fit_model"]


def __getattr__(name):
    if name == "DataStorage":
        from enefit_prosumer.data import DataStorage

        return DataStorage
    if name == "FeatureEngineer":
        from enefit_prosumer.features import FeatureEngineer

        return FeatureEngineer
    if name == "fit_model":
        from enefit_prosumer.model import fit_model

        return fit_model
    raise AttributeError(name)
