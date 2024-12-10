import pandas as pd
import numpy as np
import csv
import io

from typing import Union
from PIL import Image
from sklearn.preprocessing import StandardScaler

class DataHandler:
    """
    A class for handling data loading and preprocessing from various measurement systems.
    """

    @staticmethod
    def preprocess_gocator(raw_gocator_file: bytes) -> pd.DataFrame:
        """
        Preprocesses an unstructured CSV file containing point cloud data and converts it into a structured DataFrame.

        Args:
            raw_gocator_file (bytes): The binary content of the unstructured CSV file.

        Returns:
            pd.DataFrame: A DataFrame with structured point cloud data (columns: X, Y, Z).
        """
        point_cloud_data = []
        x_coordinates = []
        in_point_cloud_section = False

        # Decode binary content and read as CSV
        decoded_file = io.TextIOWrapper(io.BytesIO(raw_gocator_file), encoding='utf-8')
        csv_reader = csv.reader(decoded_file)

        for row in csv_reader:
            if row:
                # Check if we're at the beginning of the point cloud section.
                if row[0].startswith('Y\\X'):
                    in_point_cloud_section = True
                    x_coordinates = row[1:]  # The rest of the row contains the X coordinates.
                    x_coordinates = [float(x) for x in x_coordinates if x]  # Convert to float and remove empty strings.
                    continue  # Skip to the next row after saving the X coordinates.

                # Check if we've reached the end of the point cloud section.
                if row[0] == 'End':
                    in_point_cloud_section = False

                # If we're in the point cloud section and it's not the end, process the Z coordinates.
                if in_point_cloud_section:
                    try:
                        y_coordinate = float(row[0])
                        # Iterate over each Z value, pair it with the corresponding X coordinate, and store it with the Y coordinate.
                        for x_index, z_str in enumerate(row[1:]):
                            if z_str:  # Check if the Z value is not empty.
                                z_value = float(z_str)
                                x_value = x_coordinates[x_index]
                                point_cloud_data.append([x_value, y_coordinate, z_value])
                    except ValueError:
                        # Handle potential errors in data conversion gracefully.
                        continue

        # Create a structured DataFrame from the point cloud data
        df = pd.DataFrame(point_cloud_data, columns=['X', 'Y', 'Z'])
        df = df.fillna(0)  # Replace any NaN values with 0 for completeness

        return df


    @staticmethod
    def process_thermal_data(binary_data: bytes, conversion_type: str = "temperature") -> np.ndarray:
        """
        Processes thermal data by converting binary image data to a NumPy array
        and optionally transforming it to temperature data.

        Args:
            binary_data (bytes): Binary image data.
            conversion_type (str): Type of conversion to perform. Options are:
                                - "temperature": Convert intensity to temperature.
                                - "raw": Return the raw NumPy array of intensity values.

        Returns:
            np.ndarray: Processed data as a NumPy array.
        """
        # Convert binary data to a NumPy array
        image = Image.open(io.BytesIO(binary_data))
        intensity_array = np.array(image, dtype=np.float32)

        if conversion_type == "temperature":
            # Perform intensity-to-temperature conversion
            Emiss = 0.31
            TRefl = 301
            TAtm = 298.15
            Tau = 1.0
            TransmissionExtOptics = 1.0
            R, B, F = 24805.7, 1549.7, 1.05
            J1, J0 = 32.0948, 19915

            K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
            r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))
            r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))
            r3 = ((1 - TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R / (np.exp(B / TRefl) - F))
            K2 = r1 + r2 + r3

            data_obj_signal = (intensity_array - J0) / J1
            temperature_array = (B / np.log(R / ((K1 * data_obj_signal) - K2) + F)) - 273.15
            return temperature_array

        elif conversion_type == "raw":
            # Return raw intensity values
            return intensity_array

        else:
            raise ValueError("Invalid conversion type. Choose 'temperature' or 'raw'.")
        

    @staticmethod
    def preprocess_timeseries(
        data: pd.DataFrame,
        features: list,
        interval_seconds: int = 83,
        resample_frequency: str = '1S',
        standardize: bool = True,
        step_no_filter: Union[int, float] = 9.0
    ) -> np.ndarray:
        """
        Preprocesses and segments time-series data from a process recording.

        Args:
            data (pd.DataFrame): The input time-series data as a DataFrame.
            features (list): List of features to filter and process.
            interval_seconds (int): Duration of intervals for partitioning.
            resample_frequency (str): Resampling frequency (e.g., '1S' for 1 second).
            standardize (bool): Whether to standardize the features.
            step_no_filter (Union[int, float]): Step number to filter by.

        Returns:
            np.ndarray: Preprocessed and segmented data as a NumPy array of shape 
                        (num_segments, interval_seconds, num_features).
        """
        # Ensure 'Time' column is a datetime index
        data['Time'] = pd.to_datetime(data['Time'], dayfirst=True)
        data.set_index('Time', inplace=True)

        # Filter columns
        selected_columns = ['Step_No'] + features
        if 'Step_No' not in data.columns:
            raise ValueError("The DataFrame must include a 'Step_No' column for filtering.")

        filtered_data = data[selected_columns]

        # Resample to a fixed frequency and interpolate missing values
        resampled_data = filtered_data.resample(resample_frequency).mean()
        interpolated_data = resampled_data.interpolate(method='time')

        # Filter by specific Step_No
        if step_no_filter is not None:
            filtered_data = interpolated_data[interpolated_data['Step_No'] == step_no_filter]

        # Drop the 'Step_No' column as it is not used for further processing
        filtered_data = filtered_data.drop(columns=['Step_No'])

        # Standardize the features if required
        if standardize:
            scaler = StandardScaler(with_std=False)
            filtered_data[features] = scaler.fit_transform(filtered_data[features])

        # Segment the data into fixed intervals
        num_rows = len(filtered_data)
        
        segmented_data = []
        for start in range(0, num_rows, interval_seconds):
            end = start + interval_seconds
            segment = filtered_data.iloc[start:end]

            # Pad with the last row if the segment is shorter than the required length
            if len(segment) < interval_seconds:
                padding_length = interval_seconds - len(segment)
                padding = pd.DataFrame([segment.iloc[-1]] * padding_length, columns=segment.columns)
                segment = pd.concat([segment, padding], ignore_index=True)

            segmented_data.append(segment.to_numpy())

        # Convert the list of segments into a 3D NumPy array
        array_data = np.stack(segmented_data, axis=0)

        return array_data
