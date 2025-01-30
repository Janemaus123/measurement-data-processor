import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List
from baseclass import MeasurementProcessor


class FlirThermalProcessor(MeasurementProcessor):
    """
    Processor for FLIR Thermal Camera data.
    """

    def __init__(self, input_path: str, output_dir: str = "visualizations/thermal"):
        """
        Initialize the FLIR Thermal Processor.

        Args:
            input_path (str): Directory containing thermal image slices.
            output_dir (str): Directory where visualizations will be saved.
        """
        # FLIR calibration constants
        R, B, F = 24805.7, 1549.7, 1.05
        J1, J0 = 32.0948, 19915
        DEFAULT_EMISSIVITY = 0.31
        self.Emiss = self.DEFAULT_EMISSIVITY
        self.input_path = input_path
        self.output_dir = output_dir
        self._ensure_directory(self.output_dir)


    def _ensure_directory(self, path: str) -> None:
        """Ensures that the given directory exists."""
        os.makedirs(path, exist_ok=True)

    def _convert_to_temperature(self, intensity_array: np.ndarray) -> np.ndarray:
        """
        Converts raw FLIR intensity data to temperature values in Celsius.

        Args:
            intensity_array (np.ndarray): Raw intensity values.

        Returns:
            np.ndarray: Temperature values in Celsius.
        """
        TRefl, TAtm, Tau, TransmissionExtOptics = 301, 298.15, 1.0, 1.0
        K1 = 1 / (Tau * self.Emiss * TransmissionExtOptics)
        r1 = ((1 - self.Emiss) / self.Emiss) * (self.R / (np.exp(self.B / TRefl) - self.F))
        r2 = ((1 - Tau) / (self.Emiss * Tau)) * (self.R / (np.exp(self.B / TAtm) - self.F))
        r3 = ((1 - TransmissionExtOptics) / (self.Emiss * Tau * TransmissionExtOptics)) * (self.R / (np.exp(self.B / TRefl) - self.F))
        K2 = r1 + r2 + r3

        data_obj_signal = (intensity_array - self.J0) / self.J1
        return (self.B / np.log(self.R / ((K1 * data_obj_signal) - K2) + self.F)) - 273.15

    def _load_image(self, file_path: str, conversion_type: str) -> np.ndarray:
        """
        Loads a single thermal image and converts it to temperature if required.

        Args:
            file_path (str): Path to the image file.
            conversion_type (str): "temperature" or "raw".

        Returns:
            np.ndarray: Processed image array.
        """
        image = Image.open(file_path)
        intensity_array = np.array(image, dtype=np.float32)

        if conversion_type == "temperature":
            return self._convert_to_temperature(intensity_array)
        return intensity_array

    def _load_slices(self, conversion_type: str = "temperature") -> np.ndarray:
        """
        Loads all thermal slices in the directory and stacks them into a 3D NumPy array.

        Args:
            conversion_type (str): "temperature" or "raw".

        Returns:
            np.ndarray: 3D NumPy array of stacked thermal images.
        """
        slice_files = sorted(
            [f for f in os.listdir(self.input_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))],
            key=lambda x: int("".join(filter(str.isdigit, x)))  # Sort by numeric value in filename
        )

        if not slice_files:
            raise FileNotFoundError(f"No thermal images found in directory: {self.input_path}")

        slices = [self._load_image(os.path.join(self.input_path, file), conversion_type) for file in slice_files]
        return np.stack(slices, axis=0)

    def preprocess_data(self, conversion_type: str = "temperature") -> np.ndarray:
        """
        Preprocesses all slices in the input directory and returns a stacked 3D NumPy array.

        Args:
            conversion_type (str): "temperature" or "raw".

        Returns:
            np.ndarray: Preprocessed 3D thermal data.
        """
        return self._load_slices(conversion_type)

    def _extract_gradients(self, stack: np.ndarray) -> Dict[str, float]:
        """
        Computes gradients along each axis.

        Args:
            stack (np.ndarray): 3D thermal data stack.

        Returns:
            dict: Extracted gradient features.
        """
        gradient_z, gradient_y, gradient_x = np.gradient(stack, axis=(0, 1, 2))
        return {
            "gradient_x_mean": np.mean(gradient_x),
            "gradient_y_mean": np.mean(gradient_y),
            "gradient_z_mean": np.mean(gradient_z),
            "gradient_x_std": np.std(gradient_x),
            "gradient_y_std": np.std(gradient_y),
            "gradient_z_std": np.std(gradient_z),
        }

    def extract_features(self, stack: np.ndarray) -> Dict[str, float]:
        """
        Extracts statistical and gradient-based features from thermal data.

        Args:
            stack (np.ndarray): 3D thermal data stack.

        Returns:
            dict: Extracted features.
        """
        if stack.ndim != 3:
            raise ValueError("Input stack must be a 3D NumPy array with shape (num_layers, height, width).")

        features = {
            "mean": np.mean(stack),
            "max": np.max(stack),
            "min": np.min(stack),
            "std": np.std(stack),
        }
        features.update(self._extract_gradients(stack))
        return features

    def _generate_3d_visualization(self, processed_data: np.ndarray) -> None:
        """
        Generates a 3D visualization of the stacked thermal data.

        Args:
            processed_data (np.ndarray): Preprocessed thermal data stack (3D NumPy array).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        z_layers, y_size, x_size = processed_data.shape
        X, Y = np.meshgrid(range(x_size), range(y_size))

        for z in range(z_layers):
            ax.plot_surface(X, Y, np.full_like(X, z), facecolors=plt.cm.hot(processed_data[z, :, :]), rstride=1, cstride=1, antialiased=True)

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Layer")
        ax.set_title("3D Thermal Stacked Visualization")

        output_file = os.path.join(self.output_dir, "thermal_3d_stack.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved 3D stacked thermal visualization: {output_file}")

    def visualize_data(self, processed_data: np.ndarray) -> None:
        """
        Visualizes slices and stacked 3D thermal data.

        Args:
            processed_data (np.ndarray): Preprocessed thermal data stack (3D NumPy array).
        """
        if processed_data.ndim != 3:
            raise ValueError("Input data must be a 3D array for visualization.")

        self._ensure_directory(self.output_dir)

        for i, slice_data in enumerate(processed_data):
            plt.figure(figsize=(8, 6))
            plt.imshow(slice_data, cmap="hot", interpolation="nearest")
            plt.colorbar(label="Temperature (°C)")
            plt.title(f"Thermal Slice {i+1}")
            output_file = os.path.join(self.output_dir, f"slice_{i+1}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved slice visualization: {output_file}")

        self._generate_3d_visualization(processed_data)
