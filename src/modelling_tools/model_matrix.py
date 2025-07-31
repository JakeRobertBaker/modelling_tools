from typing import Union, List, Dict, Any
import pandas as pd
import numpy as np
from pandas import Timedelta, Timestamp

from modelling_tools import seasonality

# JSONSerializable = Union[str, int, float, bool, None, Dict[str, "JSONSerializable"], List["JSONSerializable"]]


class ModelMatrix:
    def __init__(
        self,
        datetime_col: str = "ds",
        other_features: Union[
            List[str],  # List of feature names
            Dict[str, Dict[str, Any]],  # Dict of feature names to dicts of attribute names and values
        ] = [],
    ):
        """Initialise a model matrix with.
        Args:
            datetime_col (str): Name of the date column in the DataFrame
            other_features (list[str] | dict[str, dict]): List of other features to include or a dictionary
            with feature names as keys and additional attributes as values.
        """
        self.datetime_col = datetime_col
        self.features: Dict[str, Dict[str, Any]] = {datetime_col: {"datetime": True}}

        # by default features are input features
        if isinstance(other_features, list):
            self.features.update({feature: {"derived_feature": False} for feature in other_features})
        elif isinstance(other_features, dict):
            self.features.update({feature: {"derived_feature": False, **attr} for feature, attr in other_features.items()})

        self.data_assigned = False

    def get_features(self, attribute_filters: Union[List[str], Dict[str, List]] = None) -> Dict[str, Dict[str, Any]]:
        """Get the features of the model matrix."""
        if attribute_filters is None:
            return self.features

        if not isinstance(attribute_filters, (list, dict)):
            raise ValueError("attribute_filters must be a List[str] or  Dict[str, List].")

        # validate attribute_filters
        if isinstance(attribute_filters, list):
            for filter_attr in attribute_filters:
                if not isinstance(filter_attr, str):
                    raise ValueError(f"Filter attribute '{filter_attr}' must be a string.")
        if isinstance(attribute_filters, dict):
            for filter_attr, filter_attr_values in attribute_filters.items():
                if not isinstance(filter_attr_values, list):
                    raise ValueError(f"Values for attribute '{filter_attr}' must be a list, got {type(filter_attr_values)}.")

        relevant_features = {
            feature: attr_dict
            for feature, attr_dict in self.features.items()
            if all((filter_attr_name in attr_dict for filter_attr_name in attribute_filters))
        }

        if isinstance(attribute_filters, list):
            return relevant_features

        if isinstance(attribute_filters, dict):
            return {
                feature: attr_dict
                for feature, attr_dict in relevant_features
                if all(
                    (
                        attr_dict[filter_attr_name] in filter_attr_values
                        for filter_attr_name, filter_attr_values in attribute_filters.items()
                    )
                )
            }

    def add_feature(
        self,
        feature_name: str,
        attributes: Dict[str, Any] = None,
    ) -> None:
        """Add a feature to the model matrix."""
        if attributes is None:
            attributes = {}
        if feature_name in self.features:
            raise ValueError(f"Feature '{feature_name}' already exists in the model matrix.")
        self.features[feature_name] = attributes

    def add_seasonality_feature(self, period: float, fourier_order: int, seasonality_name: str, time_col: str = None) -> None:
        """Add a derived seasonality feature to the model matrix.

        Args:
            period (float): Period of the seasonality (in days)
            fourier_order (int): Order of the Fourier series to use
            seasonality_name (str): Name for the seasonality column. Defaults to None.
        """
        time_col = time_col if time_col else self.datetime_col

        self.add_feature(
            seasonality_name,
            {"derived_feature": "seasonality", "period": period, "fourier_order": fourier_order, "time_col": time_col},
        )

    def add_daily_seasonality(self, time_col: str, fourier_order: int = 4, seasonality_name: str = "seasonality_daily"):
        self.add_seasonality_feature(period=1, fourier_order=fourier_order, seasonality_name=seasonality_name, time_col=time_col)

    def add_weekly_seasonality(self, time_col: str, fourier_order: int = 3, seasonality_name: str = "seasonality_weekly"):
        self.add_seasonality_feature(period=7, fourier_order=fourier_order, seasonality_name=seasonality_name, time_col=time_col)

    def add_yearly_seasonality(self, time_col: str, fourier_order: int = 10, seasonality_name: str = "seasonality_yearly"):
        self.add_seasonality_feature(period=365.25, fourier_order=fourier_order, seasonality_name=seasonality_name, time_col=time_col)

    def validate_matrix(self, df: pd.DataFrame):
        """Validate the model matrix against the DataFrame."""
        if self.datetime_col not in df.columns:
            raise ValueError(f"Date column '{self.datetime_col}' not found in DataFrame.")

        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"Column '{feature}' not found in DataFrame.")

    def assign_data(self, df: pd.DataFrame, auto_scale_time: bool = True):
        """Assign data to the model matrix."""
        self.validate_matrix(df)
        self.data = df.copy()
        self.data_assigned = True

        if auto_scale_time:
            time_start: Timestamp = self.data[self.datetime_col].min()
            time_end: Timestamp = self.data[self.datetime_col].max()
            datetime_scale: Timedelta = time_end - time_start
            datetime_shift = time_start
            self.features[self.datetime_col]["transform"] = {"type": "linear", "scale": datetime_scale, "shift": datetime_shift}

    def transform_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame into a model matrix."""
        if not self.data_assigned:
            raise ValueError("Data has not been assigned to the model matrix. Call assign_data() first.")

        # Ensure the DataFrame has the datetime column
        if self.datetime_col not in df.columns:
            raise ValueError(f"Date column '{self.datetime_col}' not found in DataFrame.")

        transformed_df = pd.DataFrame(index=df.index)

        input_features = {col: attrs for col, attrs in self.features.items() if not attrs.get("derived_feature", False)}

        # transform input features
        for feature, attrs in input_features.items():
            if attrs.get("transform"):
                transformed_df[feature] = self._transform_feature(df[feature], attrs["transform"])
            else:
                transformed_df[feature] = df[feature]

        # get derived features
        derived_features = {col: attrs for col, attrs in self.features.items() if attrs.get("derived_feature", False)}
        for feature, attrs in derived_features.items():
            derived_feature_df = self._derive_features(df, feature, attrs)
            transformed_df = pd.concat([transformed_df, derived_feature_df], axis=1)

        return transformed_df

    def _derive_features(self, df, feature: str, attr: Dict[str, Any]) -> pd.DataFrame:
        if self.datetime_col not in df.columns:
            raise ValueError(f"Date column '{self.datetime_col}' not found in DataFrame.")

        if attr["derived_feature"] == "seasonality":
            seasonality_df = seasonality.generate_seasonality(
                df,
                date_col=attr["time_col"],
                period=attr["period"],
                fourier_order=attr["fourier_order"],
                seasonality_name=feature,
            )
            return seasonality_df
        else:
            raise ValueError(f"Unsupported derived feature type: {attr['derived_feature']} for feature '{feature}'.")

    def _transform_feature(self, data: Union[pd.Series, np.ndarray], transform: Dict[str, Any]) -> Union[pd.Series, np.ndarray]:
        """Apply a transformation to a pandas Series or numpy array."""
        if transform["type"] == "linear":
            return (data - transform["shift"]) / transform["scale"]
        else:
            raise ValueError(f"Unsupported transformation type: {transform['type']}")
