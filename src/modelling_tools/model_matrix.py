from typing import Union, List, Dict, Any
import pandas as pd
import numpy as np
from pandas import Timedelta, Timestamp

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

        if isinstance(other_features, list):
            self.features.update({col: {} for col in other_features})
        elif isinstance(other_features, dict):
            self.features.update(other_features)

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

        relevant_columns = {
            col
            for col, attr_dict in self.features.items()
            if all((filter_attr_name in attr_dict for filter_attr_name in attribute_filters))
        }

        if isinstance(attribute_filters, list):
            return relevant_columns

        if isinstance(attribute_filters, dict):
            return {
                col: attr_dict
                for col, attr_dict in relevant_columns
                if all(
                    (
                        attr_dict[filter_attr_name] in filter_attr_values
                        for filter_attr_name, filter_attr_values in attribute_filters.items()
                    )
                )
            }

    def validate_matrix(self, df: pd.DataFrame):
        """Validate the model matrix against the DataFrame."""
        if self.datetime_col not in df.columns:
            raise ValueError(f"Date column '{self.datetime_col}' not found in DataFrame.")

        for col in self.features:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

    def assign_data(self, df: pd.DataFrame, auto_scale_time: bool = True):
        """Assign data to the model matrix."""
        self.validate_matrix(df)
        self.data = df.copy()
        self.data_assigned = True

        if auto_scale_time:
            time_start: Timestamp = self.data[self.datetime_col].min()
            time_end: Timestamp = self.data[self.datetime_col].max()
            datetime_scale: Timedelta = time_start - time_end
            datetime_shift = time_start
            self.features[self.datetime_col]["transform"] = {"type": "linear", "scale": datetime_scale, "shift": datetime_shift}

    def transform_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame into a model matrix."""
        if not self.data_assigned:
            raise ValueError("Data has not been assigned to the model matrix. Call assign_data() first.")

        # Ensure the DataFrame has the datetime column
        if self.datetime_col not in df.columns:
            raise ValueError(f"Date column '{self.datetime_col}' not found in DataFrame.")

        # Create a copy of the DataFrame to avoid modifying the original
        transformed_df = df.copy()

        # Apply transformations based on features
        for col, attrs in self.features.items():
            if attrs.get("transform"):
                transformed_df[col] = self.apply_transformation(transformed_df[col], attrs["transform"])

        return transformed_df

    def apply_transformation(self, data: Union[pd.Series, np.ndarray], transform: Dict[str, Any]) -> Union[pd.Series, np.ndarray]:
        """Apply a transformation to a pandas Series or numpy array."""
        if transform["type"] == "linear":
            return (data - transform["shift"]) / transform["scale"]
        else:
            raise ValueError(f"Unsupported transformation type: {transform['type']}")