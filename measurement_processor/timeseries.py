import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from typing import Union
from baseclass import MeasurementProcessor  # Abstract base class


class TimeseriesProcessor(MeasurementProcessor):
    """
    Processor for Time-Series Data.
    """

    def __init__(self, input_path: str, output_dir: str = "visualizations/timeseries", interval_seconds: int = 83):
        """
        Constructor for the Timeseries Processor class.

        Args:
            input_path (str): Path to the input time-series data file (CSV).
            output_dir (str): Directory where visualizations will be saved.
            interval_seconds (int): Length of each time-series segment in seconds.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        features: list, 
        resample_frequency: str = '1s', 
        standardize: bool = True, 
        step_no_filter: Union[int, float] = 9.0
    ) -> np.ndarray:
        """
        Preprocesses and segments time-series data into fixed intervals.

        Args:
            data (pd.DataFrame): Input time-series data.
            features (list): List of features to process.
            resample_frequency (str): Resampling frequency (e.g., '1s').
            standardize (bool): Whether to standardize the data.
            step_no_filter (Union[int, float]): Step number to filter by.

        Returns:
            np.ndarray: Segmented time-series data as a 3D NumPy array.
        """
        # Ensure 'Time' column is in timedelta format
        data['Time'] = pd.to_timedelta(data['Time'])

        # Set the Time column as the index
        data.set_index('Time', inplace=True)

        # Filter columns
        selected_columns = ["Step_No."] + features
        if 'Step_No.' not in data.columns:
            raise ValueError("The DataFrame must include a 'Step_No.' column for filtering.")

        filtered_data = data[selected_columns]

        # Resample and interpolate missing values
        resampled_data = filtered_data.resample(resample_frequency).mean()
        interpolated_data = resampled_data.interpolate(method='time')

        # Filter by Step_No.
        if step_no_filter is not None:
            filtered_data = interpolated_data[interpolated_data['Step_No.'] == step_no_filter]

        # Drop Step_No and calculate Run_IDs
        filtered_data = filtered_data.drop(columns=['Step_No.'])
        time_deltas = filtered_data.index.to_series().diff().dt.total_seconds()
        filtered_data['Run_ID'] = (time_deltas > 10).cumsum()

        segmented_data = []

        # Process each Run_ID separately
        for run_id, run_data in filtered_data.groupby('Run_ID'):
            if standardize:
                scaler = StandardScaler(with_std=False)
                run_data_scaled = pd.DataFrame(
                    scaler.fit_transform(run_data),
                    columns=run_data.columns,
                    index=run_data.index
                )
            else:
                run_data_scaled = run_data

            # Segment the standardized data
            num_rows = len(run_data_scaled)
            for start in range(0, num_rows, self.interval_seconds):
                end = start + self.interval_seconds
                segment = run_data_scaled.iloc[start:end]

                # Pad segments if needed
                if len(segment) < self.interval_seconds:
                    padding_length = self.interval_seconds - len(segment)
                    padding = pd.DataFrame([segment.iloc[-1]] * padding_length, columns=segment.columns)
                    segment = pd.concat([segment, padding], ignore_index=True)

                segmented_data.append(segment.to_numpy())

        return np.stack(segmented_data, axis=0)[:, :, :-1]  # Exclude Run_ID from features

    def extract_features(self, segments: np.ndarray, feature_columns: list) -> list:
        """
        Extracts statistical and signal-based features from segmented time-series data.

        Args:
            segments (np.ndarray): 3D array of shape (num_segments, interval_seconds, num_features).
            feature_columns (list): List of feature names.

        Returns:
            list: A list of dictionaries, each containing features for a segment.
        """
        extracted_features = []

        for segment in segments:
            segment_features = {}
            for i, feature_name in enumerate(feature_columns):
                feature_data = segment[:, i]

                # Statistical features
                segment_features[f'{feature_name}_mean'] = np.mean(feature_data)
                segment_features[f'{feature_name}_std'] = np.std(feature_data)
                segment_features[f'{feature_name}_min'] = np.min(feature_data)
                segment_features[f'{feature_name}_max'] = np.max(feature_data)
                segment_features[f'{feature_name}_skew'] = skew(feature_data)
                segment_features[f'{feature_name}_kurtosis'] = kurtosis(feature_data)

                # Signal-based features
                peaks, _ = find_peaks(feature_data)
                segment_features[f'{feature_name}_num_peaks'] = len(peaks)
                autocorr = np.correlate(
                    feature_data - np.mean(feature_data),
                    feature_data - np.mean(feature_data),
                    mode='full'
                )
                segment_features[f'{feature_name}_autocorr'] = autocorr[autocorr.size // 2]

            extracted_features.append(segment_features)

        return extracted_features

    def visualize_data(self, 
        timeseries_array,  
        feature_names, 
        output_folder="visualizations/timeseries"
    ):
        """
        Plots all time series from a 3D NumPy array with fixed y-axis range and tick intervals.

        Args:
            timeseries_array (np.ndarray): A 3D array with shape (num_segments, interval_seconds, num_features).
            feature_names (list): List of feature names corresponding to the last dimension of the array.
            output_folder (str): Path to the folder where plots are saved.

        Returns:
            None
        """
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Fixed parameters for y-axis
        y_min, y_max = -3, 3
        y_ticks_interval = 0.5

        # x-axis duration (83 seconds assumed)
        x_ticks_interval = 10  # For example, one tick every 10 seconds

        # Iterate over each segment (Run_ID)
        for run_id, segment in enumerate(timeseries_array, start=1):
            plt.figure(figsize=(12, 6))

            # Plot each feature in the segment
            for feature_index, feature_name in enumerate(feature_names):
                plt.plot(segment[:, feature_index], label=feature_name, linewidth=1.5)

            # Add grid lines
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Set labels and title
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Sensor Values', fontsize=12)

            # Set fixed y-axis limits and tick intervals
            plt.ylim(y_min, y_max)
            plt.gca().yaxis.set_major_locator(plt.MultipleLocator(y_ticks_interval))

            # Set x-axis limits and ticks
            plt.xlim(0, 83)  # Assuming 83 seconds length
            plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_ticks_interval))

            # Add title and legend
            plt.title(f"Time Series Plot, Run ID: {run_id}", fontsize=14)
            plt.legend(loc="upper left")

            # Format filename
            output_filename = f"run_{run_id}.png"
            output_path = os.path.join(output_folder, output_filename)

            # Save plot
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved at {output_path}")

            # Close plot to free memory
            plt.close()
