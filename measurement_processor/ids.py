import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from measurement_processor import MeasurementProcessor


class OpticalProcessor(MeasurementProcessor):
    """
    Processor for IDS Optical Images.
    """

    CROP_SIZE = 3072  # Fixed crop size
    TILE_SIZE = 256   # Tile size

    def __init__(self, input_dir: str, output_dir: str = "visualizations/ids"):
        """
        Initialize the IDS Optical Processor.

        Args:
            input_dir (str): Directory containing optical images.
            output_dir (str): Directory for saving visualizations.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_image(self, file_path: str) -> np.ndarray:
        """
        Load an image and apply optional random horizontal flipping.

        Args:
            file_path (str): Path to the image file.
            train_mode (bool): Enables random augmentation.

        Returns:
            np.ndarray: Loaded and optionally flipped image.
        """
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

        return image

    def _validate_image_size(self, image: np.ndarray):
        """
        Validates that the image size is at least 3072x3072.

        Args:
            image (np.ndarray): Image to validate.

        Raises:
            ValueError: If image size is smaller than required crop size.
        """
        h, w, _ = image.shape
        if h < self.CROP_SIZE or w < self.CROP_SIZE:
            raise ValueError(f"Image size must be at least {self.CROP_SIZE}x{self.CROP_SIZE}, but got {h}x{w}")

    def _crop_and_tile(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Crop the image to 3072x3072 and split it into 256x256 tiles.

        Args:
            image (np.ndarray): Input image.

        Returns:
            List[np.ndarray]: List of 256x256 tiles.
        """
        self._validate_image_size(image)

        # Central crop
        h, w, _ = image.shape
        start_h, start_w = (h - self.CROP_SIZE) // 2, (w - self.CROP_SIZE) // 2
        cropped_image = image[start_h:start_h + self.CROP_SIZE, start_w:start_w + self.CROP_SIZE]

        # Tile the cropped image into 256x256 patches
        tiles = [
            cropped_image[i:i + self.TILE_SIZE, j:j + self.TILE_SIZE]
            for i in range(0, self.CROP_SIZE, self.TILE_SIZE)
            for j in range(0, self.CROP_SIZE, self.TILE_SIZE)
        ]

        return tiles  # 144 tiles (12x12 grid)

    def preprocess_data(self, file_path: str, train_mode: bool = False) -> List[np.ndarray]:
        """
        Preprocess an optical image by loading, cropping, and tiling it.

        Args:
            file_path (str): Path to the image file.
            train_mode (bool): Enables random augmentation.

        Returns:
            List[np.ndarray]: List of preprocessed 256x256 image tiles.
        """
        image = self._load_image(file_path, train_mode)
        return self._crop_and_tile(image)

    def _compute_color_statistics(self, tiles: List[np.ndarray]) -> Dict[str, float]:
        """
        Computes mean and standard deviation of RGB values across tiles.

        Args:
            tiles (List[np.ndarray]): List of 256x256 image tiles.

        Returns:
            Dict[str, float]: Dictionary containing extracted features.
        """
        tile_means = np.array([np.mean(tile, axis=(0, 1)) for tile in tiles])  # Per-tile mean RGB
        tile_stds = np.array([np.std(tile, axis=(0, 1)) for tile in tiles])    # Per-tile std RGB

        return {
            "num_tiles": len(tiles),
            "mean_red": np.mean(tile_means[:, 0]),
            "mean_green": np.mean(tile_means[:, 1]),
            "mean_blue": np.mean(tile_means[:, 2]),
            "std_red": np.mean(tile_stds[:, 0]),
            "std_green": np.mean(tile_stds[:, 1]),
            "std_blue": np.mean(tile_stds[:, 2]),
        }

    def extract_features(self, tiles: List[np.ndarray]) -> Dict[str, float]:
        """
        Extract features from the processed tiles.

        Args:
            tiles (List[np.ndarray]): List of 256x256 image tiles.

        Returns:
            Dict[str, float]: Extracted features.
        """
        return self._compute_color_statistics(tiles)

    def visualize_data(self, tiles: List[np.ndarray], output_path: str):
        """
        Visualize and save the tiled images.

        Args:
            tiles (List[np.ndarray]): List of 256x256 image tiles.
            output_path (str): Directory for saving visualizations.
        """
        os.makedirs(output_path, exist_ok=True)

        fig, axes = plt.subplots(12, 12, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            ax.imshow(tiles[i])
            ax.axis("off")

        plt.suptitle("Tiled Optical Image", fontsize=16)
        save_path = os.path.join(output_path, "tiled_image.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {save_path}")
