import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

class Preprocessor:
    """
    A class for preprocessing and feature extraction from time-series data.
    """

    @staticmethod
    def extract_features_from_segments(segments: np.ndarray, feature_columns: list) -> list:
        """
        Extracts statistical and signal-based features from each time-series segment.

        Args:
            segments (np.ndarray): 3D NumPy array of shape (num_segments, interval_seconds, num_features).
            feature_columns (list): List of feature names corresponding to the columns in the input data.

        Returns:
            list: A list of dictionaries, where each dictionary contains features for one segment.
        """
        extracted_features = []

        for segment in segments:
            segment_features = {}
            for i, feature_name in enumerate(feature_columns):
                # Extract time-series data for the current feature
                feature_data = segment[:, i]

                # Calculate statistical features
                segment_features[f'{feature_name}_mean'] = np.mean(feature_data)
                segment_features[f'{feature_name}_std'] = np.std(feature_data)
                segment_features[f'{feature_name}_min'] = np.min(feature_data)
                segment_features[f'{feature_name}_max'] = np.max(feature_data)
                segment_features[f'{feature_name}_skew'] = skew(feature_data)
                segment_features[f'{feature_name}_kurtosis'] = kurtosis(feature_data)

                # Calculate signal-based features (e.g., peaks, autocorrelation)
                peaks, _ = find_peaks(feature_data)
                segment_features[f'{feature_name}_num_peaks'] = len(peaks)
                autocorr = np.correlate(
                    feature_data - np.mean(feature_data),
                    feature_data - np.mean(feature_data),
                    mode='full'
                )
                segment_features[f'{feature_name}_autocorr'] = autocorr[autocorr.size // 2]

            # Append the extracted features for this segment
            extracted_features.append(segment_features)

        return extracted_features
    
    
