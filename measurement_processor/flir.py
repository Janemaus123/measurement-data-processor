import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from baseclass import MeasurementProcessor  # Abstract base class
import io

class FlirThermalProcessor(MeasurementProcessor):
    """
    Processor for FLIR Thermal Camera data.
    """

    def __init__(self, input_path: str, output_dir: str = "visualizations/thermal"):
        """
        Constructor for the FLIR Thermal Processor class.

        Args:
            input_path (str): Path to the input thermal files.
            output_dir (str): Directory where visualizations will be saved.
        """
        self.Emiss = 0.31
        self.input_path = input_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_data(self, raw_data: bytes, conversion_type: str = "temperature") -> np.ndarray:
        """
        Preprocess raw thermal data and convert to a NumPy array.

        Args:
            raw_data (bytes): Binary input data.
            conversion_type (str): "temperature" or "raw" for processing type.

        Returns:
            np.ndarray: Preprocessed thermal data as a NumPy array.
        """
        with open(self.input_path, "rb") as file:
            raw_data = file.read()
        
        image = Image.open(io.BytesIO(raw_data))
        intensity_array = np.array(image, dtype=np.float32)

        if conversion_type == "temperature":
            # Perform intensity-to-temperature conversion
            TRefl = 301
            TAtm = 298.15
            Tau = 1.0
            TransmissionExtOptics = 1.0
            R, B, F = 24805.7, 1549.7, 1.05
            J1, J0 = 32.0948, 19915

            K1 = 1 / (Tau * self.Emiss * TransmissionExtOptics)
            r1 = ((1 - self.Emiss) / self.Emiss) * (R / (np.exp(B / TRefl) - F))
            r2 = ((1 - Tau) / (self.Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))
            r3 = ((1 - TransmissionExtOptics) / (self.Emiss * Tau * TransmissionExtOptics)) * (R / (np.exp(B / TRefl) - F))
            K2 = r1 + r2 + r3

            data_obj_signal = (intensity_array - J0) / J1
            temperature_array = (B / np.log(R / ((K1 * data_obj_signal) - K2) + F)) - 273.15
            return temperature_array

        elif conversion_type == "raw":
            return intensity_array

        else:
            raise ValueError("Invalid conversion type. Choose 'temperature' or 'raw'.")

    def extract_features(self, stack: np.ndarray) -> dict:
        """
        Extracts features from a stack of thermal data.

        Args:
            stack (np.ndarray): 3D thermal data stack.

        Returns:
            dict: Extracted features.
        """
        try:
            if stack.ndim != 3:
                raise ValueError("Input stack must be a 3D NumPy array with shape (num_layers, height, width).")

            # Compute gradients along each axis
            gradient_z, gradient_y, gradient_x = np.gradient(stack, axis=(0, 1, 2))

            # Extract statistical features
            features = {
                "mean": np.mean(stack),
                "max": np.max(stack),
                "min": np.min(stack),
                "std": np.std(stack),
                "gradient_x_mean": np.mean(gradient_x),
                "gradient_y_mean": np.mean(gradient_y),
                "gradient_z_mean": np.mean(gradient_z),
                "gradient_x_std": np.std(gradient_x),
                "gradient_y_std": np.std(gradient_y),
                "gradient_z_std": np.std(gradient_z),
            }

            return features

        except Exception as e:
            print(f"Error during thermal data preprocessing: {e}")
            return None

    def visualize_data(self, processed_data: np.ndarray, output_path: str = None):
        """
        Visualize slices and stacked 3D thermal data.

        Args:
            processed_data (np.ndarray): Preprocessed thermal data stack (3D NumPy array).
            output_path (str): Directory to save visualizations.
        """
        if processed_data.ndim != 3:
            raise ValueError("Input data must be a 3D array for visualization.")

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Display individual slices
        num_slices = processed_data.shape[0]
        for i in range(num_slices):
            plt.figure(figsize=(8, 6))
            plt.imshow(processed_data[i, :, :], cmap='hot', interpolation='nearest')
            plt.colorbar(label="Temperature (°C)")
            plt.title(f"Thermal Slice {i+1}")
            output_file = os.path.join(self.output_dir, f"slice_{i+1}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved slice visualization: {output_file}")
            plt.close()

        # 2. 3D Visualization of Stacked Thermal Data
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Generate X, Y, Z coordinates
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
        print(f"Saved 3D stacked thermal visualization: {output_file}")
        plt.close()
