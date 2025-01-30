import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from typing import List, Union

class TimeseriesProcessor:
    """
    Processor for Time-Series Data with segmentation, preprocessing, and visualization capabilities.
    """
    def __init__(self, input_path: str, output_dir: str = "visualizations/timeseries", interval_seconds: int = 83):
        """
        Initialize the time-series processor.

        Args:
            input_path (str): Path to the input CSV file.
            output_dir (str): Directory for saving visualizations.
            interval_seconds (int): Length of each time-series segment in seconds.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        Y_MIN, Y_MAX = -3, 3
        Y_TICK_INTERVAL = 0.5
        X_TICK_INTERVAL = 10  # Tick every 10 seconds
        FIGURE_SIZE = (12, 6)
        os.makedirs(self.output_dir, exist_ok=True)

    def _validate_dataframe(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Ensures required columns exist in the DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            required_columns (List[str]): List of required column names.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def segment_timeseries(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Segments the time-series data into fixed-length intervals.

        Args:
            data (pd.DataFrame): Time-series data with a datetime index.

        Returns:
            pd.DataFrame: Data with a new 'Segment_ID' column.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a pandas DatetimeIndex.")

        start_time, end_time = data.index[0], data.index[-1]
        num_segments = int(np.ceil((end_time - start_time).total_seconds() / self.interval_seconds))

        # Vectorized segmentation
        segment_ids = ((data.index - start_time).total_seconds() // self.interval_seconds).astype(int)
        data['Segment_ID'] = segment_ids

        return data

    def preprocess_data(
        self,
        data: pd.DataFrame,
        features: List[str],
        resample_frequency: str = '1s',
        standardize: bool = True,
        step_no_filter: Union[int, float] = 9.0
    ) -> np.ndarray:
        """
        Preprocess and segment time-series data.

        Args:
            data (pd.DataFrame): Input data.
            features (List[str]): Features to process.
            resample_frequency (str): Frequency for resampling.
            standardize (bool): Whether to standardize.
            step_no_filter (Union[int, float]): Step number to filter.

        Returns:
            np.ndarray: Segmented data as a 3D NumPy array.
        """
        self._validate_dataframe(data, ["Time", "Step_No."] + features)

        data["Time"] = pd.to_timedelta(data["Time"])
        data.set_index("Time", inplace=True)

        # Resample and interpolate
        selected_columns = ["Step_No."] + features
        resampled_data = data[selected_columns].resample(resample_frequency).mean().interpolate(method='time')

        # Filter by step number
        if step_no_filter is not None:
            resampled_data = resampled_data[resampled_data["Step_No."] == step_no_filter]

        # Drop 'Step_No.' and compute segment IDs
        resampled_data.drop(columns=["Step_No."], inplace=True)
        resampled_data["Run_ID"] = (resampled_data.index.to_series().diff().dt.total_seconds() > 10).cumsum()

        # Segment alignment
        if resampled_data.index.to_series().diff().dt.total_seconds().max() > self.interval_seconds:
            resampled_data = self.segment_timeseries(resampled_data)

        segmented_data = self._split_and_standardize(resampled_data, features, standardize)

        return segmented_data

    def _split_and_standardize(self, data: pd.DataFrame, features: List[str], standardize: bool) -> np.ndarray:
        """
        Splits and optionally standardizes segmented time-series data.

        Args:
            data (pd.DataFrame): Preprocessed data.
            features (List[str]): Feature names.
            standardize (bool): Standardization flag.

        Returns:
            np.ndarray: Segmented 3D NumPy array.
        """
        segments = []
        scaler = StandardScaler(with_std=False) if standardize else None

        for _, segment_data in data.groupby(["Run_ID", "Segment_ID"]):
            segment_values = segment_data[features]

            if standardize:
                segment_values = pd.DataFrame(scaler.fit_transform(segment_values), columns=features)

            # Pad if necessary
            if len(segment_values) < self.interval_seconds:
                padding = pd.DataFrame([segment_values.iloc[-1]] * (self.interval_seconds - len(segment_values)), columns=features)
                segment_values = pd.concat([segment_values, padding], ignore_index=True)

            segments.append(segment_values.to_numpy())

        return np.stack(segments, axis=0)

    def extract_features(self, segments: np.ndarray, feature_columns: List[str]) -> List[dict]:
        """
        Extracts features from segmented time-series data.

        Args:
            segments (np.ndarray): Time-series segments.
            feature_columns (List[str]): Feature names.

        Returns:
            List[dict]: List of feature dictionaries.
        """
        extracted_features = []

        for segment in segments:
            features = {}
            for i, name in enumerate(feature_columns):
                feature_data = segment[:, i]
                features.update({
                    f"{name}_mean": np.mean(feature_data),
                    f"{name}_std": np.std(feature_data),
                    f"{name}_min": np.min(feature_data),
                    f"{name}_max": np.max(feature_data),
                    f"{name}_skew": skew(feature_data),
                    f"{name}_kurtosis": kurtosis(feature_data),
                    f"{name}_num_peaks": len(find_peaks(feature_data)[0]),
                    f"{name}_autocorr": np.correlate(feature_data - np.mean(feature_data), feature_data - np.mean(feature_data), mode='full')[feature_data.size // 2],
                })
            extracted_features.append(features)

        return extracted_features

    def _ensure_directory(self, path: str) -> None:
        """
        Ensures the given directory exists.

        Args:
            path (str): Path to the directory.
        """
        os.makedirs(path, exist_ok=True)

    def _plot_segment(self, segment: np.ndarray, feature_names: list, run_id: int, output_folder: str) -> None:
        """
        Plots a single time-series segment.

        Args:
            segment (np.ndarray): Time-series data for a single run.
            feature_names (list): Names of the features in the segment.
            run_id (int): Identifier for the segment being plotted.
            output_folder (str): Directory where plots will be saved.
        """
        plt.figure(figsize=self.FIGURE_SIZE)

        # Plot each feature
        for feature_index, feature_name in enumerate(feature_names):
            plt.plot(segment[:, feature_index], label=feature_name, linewidth=1.5)

        # Set labels and title
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Sensor Values", fontsize=12)
        plt.title(f"Time Series Plot, Run ID: {run_id}", fontsize=14)

        # Grid, legend, and ticks
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(loc="upper left")

        # Set fixed y-axis limits and ticks
        plt.ylim(self.Y_MIN, self.Y_MAX)
        plt.yticks(np.arange(self.Y_MIN, self.Y_MAX + self.Y_TICK_INTERVAL, self.Y_TICK_INTERVAL))

        # Set x-axis limits dynamically based on segment length
        segment_length = segment.shape[0]
        plt.xlim(0, segment_length)
        plt.xticks(np.arange(0, segment_length + 1, self.X_TICK_INTERVAL))

        # Save the plot
        output_filename = f"run_{run_id}.png"
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved at {output_path}")

        # Close to free memory
        plt.close()

    def visualize_data(self, timeseries_array: np.ndarray, feature_names: list, output_folder: str = "visualizations/timeseries") -> None:
        """
        Plots all time series from a 3D NumPy array with fixed y-axis range and tick intervals.

        Args:
            timeseries_array (np.ndarray): A 3D array with shape (num_segments, interval_seconds, num_features).
            feature_names (list): List of feature names corresponding to the last dimension of the array.
            output_folder (str): Path to the folder where plots are saved.

        Returns:
            None
        """
        self._ensure_directory(output_folder)

        # Iterate over each segment (Run_ID) and plot it
        for run_id, segment in enumerate(timeseries_array, start=1):
            self._plot_segment(segment, feature_names, run_id, output_folder)