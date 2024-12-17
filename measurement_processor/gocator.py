import pandas as pd
import numpy as np
import io
import csv
import os
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_data(self) -> pd.DataFrame:
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

        with open(self.input_path, "rb") as file:
            raw_gocator_file = file.read()

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

    def visualize_data(self, processed_data: pd.DataFrame, output_path: str = "gocator_point_cloud.png"):
        """
        Visualize 3D point cloud data as a scatter plot.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(processed_data['X'], processed_data['Y'], processed_data['Z'], c='b', s=1, alpha=0.7)
        ax.set_title("Gocator 3D Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        output_file = os.path.join(self.output_dir, output_path)
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Visualization saved at: {output_file}")

    def extract_features(self, gocator_df: pd.DataFrame, product_id: str) -> dict:
        """
        Extract features to quantify the deviation of the point cloud from an ideal rectangular volume.
        """
        if not all(col in gocator_df.columns for col in ['X', 'Y', 'Z']):
            raise ValueError("Input DataFrame must contain columns ['X', 'Y', 'Z'].")

        quantile_threshold = 0.05

        # Extract coordinates
        x = gocator_df['X'].to_numpy()
        y = gocator_df['Y'].to_numpy()
        z = gocator_df['Z'].to_numpy()

        z_threshold = np.quantile(z, quantile_threshold)
        valid_points = z > z_threshold

        x_filtered = x[valid_points]
        y_filtered = y[valid_points]
        z_filtered = z[valid_points]

        # Bounding box dimensions
        bounding_box = {
            "x_range": x_filtered.max() - x_filtered.min(),
            "y_range": y_filtered.max() - y_filtered.min(),
            "z_range": z_filtered.max(),
        }

        # Surface flatness
        flatness = {
            "z_height_mean": np.mean(z_filtered[z_filtered > np.median(z_filtered)]),
            "z_std_dev": np.std(z_filtered[z_filtered > np.median(z_filtered)]),
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

        features = {
            **bounding_box, **flatness, **volume_coverage, **point_density,
            "Product_ID": product_id
        }
        return features

