from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union

class MeasurementProcessor(ABC):
    """
    Abstract base class for processing, visualizing, and modeling measurement instrument data.
    """

    @abstractmethod
    def preprocess_data(self, raw_data) -> Union[pd.DataFrame, np.ndarray]:
        """
        Preprocess raw input data and return a cleaned DataFrame or NumPy array.
        
        Args:
            raw_data: Raw input data (e.g., file path, binary data, DataFrame).
        
        Returns:
            Processed data as a pandas DataFrame or NumPy array.
        """
        pass

    @abstractmethod
    def visualize_data(self, processed_data: Union[pd.DataFrame, np.ndarray], output_path: str):
        """
        Visualize the processed data and save the output as images.
        
        Args:
            processed_data: The data after preprocessing.
            output_path (str): Path to save visualizations.
        """
        pass
    
    @abstractmethod
    def extract_features(self, processed_data: Union[pd.DataFrame, np.ndarray], **kwargs) -> dict:
        """
        Extract meaningful features from the processed data.
        
        Args:
            processed_data: Data ready for feature extraction.
            **kwargs: Additional arguments relevant to feature extraction.
        
        Returns:
            A dictionary containing extracted features.
        """
        pass

    @abstractmethod
    def train_model(self, processed_data: Union[pd.DataFrame, np.ndarray], target_column: str, models=None) -> dict:
        """
        Train machine learning models on the processed data.
        
        Args:
            processed_data: Data ready for modeling.
            target_column (str): The target variable column name.
            models (dict): Dictionary of models to train and evaluate.
        
        Returns:
            A dictionary containing model metrics and trained models.
        """
        pass
