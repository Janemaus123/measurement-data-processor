import os
import io
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict
from measurement_processor import MeasurementProcessor


class GocatorProcessor(MeasurementProcessor):
    """
    Processor class for Gocator point cloud data.
    """

    def __init__(self, input_path: str, output_dir: str = "visualizations/gocator"):
        """
        Initialize GocatorProcessor with file paths.

        Args:
            input_path (str): Path to the Gocator input file.
            output_dir (str): Directory to save visualizations.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self._ensure_directory(self.output_dir)

    def _ensure_directory(self, path: str) -> None:
        """
        Ensures that the given directory exists.

        Args:
            path (str): Directory path.
        """
        os.makedirs(path, exist_ok=True)

    def _parse_csv_file(self) -> pd.DataFrame:
        """
        Parses the CSV file containing unstructured point cloud data.

        Returns:
            pd.DataFrame: A DataFrame with structured point cloud data (columns: X, Y, Z).
        """
        point_cloud_data = []
        x_coordinates = []
        in_point_cloud_section = False

        with open(self.input_path, "rb") as file:
            raw_gocator_file = file.read()

        # Decode binary content and read as CSV
        decoded_file = io.TextIOWrapper(io.BytesIO(raw_gocator_file), encoding="utf-8")
        csv_reader = csv.reader(decoded_file)

        for row in csv_reader:
            if not row:
                continue

            # Start of point cloud section
            if row[0].startswith("Y\\X"):
                in_point_cloud_section = True
                x_coordinates = [float(x) for x in row[1:] if x]  # Convert non-empty values to float
                continue

            # End of point cloud section
            if row[0] == "End":
                in_point_cloud_section = False
                break

            # Process Z values if inside point cloud section
            if in_point_cloud_section:
                try:
                    y_coordinate = float(row[0])
                    for x_index, z_str in enumerate(row[1:]):
                        if z_str:
                            point_cloud_data.append([x_coordinates[x_index], y_coordinate, float(z_str)])
                except ValueError:
                    continue  

        return pd.DataFrame(point_cloud_data, columns=["X", "Y", "Z"]).fillna(0)

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesses an unstructured CSV file containing point cloud data.

        Returns:
            pd.DataFrame: A structured DataFrame with columns [X, Y, Z].
        """
        return self._parse_csv_file()

    def visualize_data(self, processed_data: pd.DataFrame, output_path: str = "gocator_point_cloud.png"):
        """
        Visualize 3D point cloud data as a scatter plot.

        Args:
            processed_data (pd.DataFrame): DataFrame containing point cloud data.
            output_path (str): Output file name for visualization.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(processed_data["X"], processed_data["Y"], processed_data["Z"], c="b", s=1, alpha=0.7)
        ax.set_title("Gocator 3D Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Save the plot
        output_file = os.path.join(self.output_dir, output_path)
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Visualization saved at: {output_file}")

    def _filter_valid_points(self, gocator_df: pd.DataFrame, quantile_threshold: float = 0.05):
        """
        Filters valid points from the dataset based on Z-value threshold.

        Args:
            gocator_df (pd.DataFrame): The input point cloud DataFrame.
            quantile_threshold (float): The quantile threshold for Z-value filtering.

        Returns:
            np.ndarray: Filtered X, Y, and Z coordinates.
        """
        z = gocator_df["Z"].to_numpy()
        threshold = np.quantile(z, quantile_threshold)
        valid_points = z > threshold

        return gocator_df["X"].to_numpy()[valid_points], gocator_df["Y"].to_numpy()[valid_points], z[valid_points]

    def extract_features(self, gocator_df: pd.DataFrame, product_id: str) -> Dict[str, float]:
        """
        Extracts features to quantify the deviation of the point cloud from an ideal rectangular volume.

        Args:
            gocator_df (pd.DataFrame): The input point cloud DataFrame.
            product_id (str): Unique identifier for the product.

        Returns:
            dict: Dictionary containing extracted features.
        """
        required_columns = ["X", "Y", "Z"]
        if not all(col in gocator_df.columns for col in required_columns):
            raise ValueError(f"Input DataFrame must contain columns {required_columns}.")

        # Filter valid points
        x_filtered, y_filtered, z_filtered = self._filter_valid_points(gocator_df)

        # Bounding box dimensions
        bounding_box = {
            "x_range": x_filtered.max() - x_filtered.min(),
            "y_range": y_filtered.max() - y_filtered.min(),
            "z_range": z_filtered.max(),
        }

        # Surface flatness
        z_above_median = z_filtered[z_filtered > np.median(z_filtered)]
        flatness = {
            "z_height_mean": np.mean(z_above_median),
            "z_std_dev": np.std(z_above_median),
        }

        # Convex hull and volume coverage
        points = np.column_stack((x_filtered, y_filtered, z_filtered))
        hull = ConvexHull(points)
        convex_hull_volume = hull.volume
        ideal_volume = bounding_box["x_range"] * bounding_box["y_range"] * bounding_box["z_range"]
        volume_coverage = {
            "convex_hull_volume": convex_hull_volume,
            "ideal_volume": ideal_volume,
            "volume_ratio": convex_hull_volume / ideal_volume if ideal_volume > 0 else np.nan,
        }

        # Point density
        point_density = {
            "num_points": len(points),
            "points_per_unit_volume": len(points) / convex_hull_volume if convex_hull_volume > 0 else np.nan,
        }

        # Merge all extracted features
        features = {
            **bounding_box,
            **flatness,
            **volume_coverage,
            **point_density,
            "Product_ID": product_id,
        }

        return features
