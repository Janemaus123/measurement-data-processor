import os
import owncloud
from dotenv import load_dotenv
from measurement_processor.data_handler import DataHandler
from measurement_processor.visualizer import Visualizer
from measurement_processor.setup import FeatureExtractor
import pandas as pd


def process_gocator_files_in_products():
    """
    Processes Gocator CSV files from a structured directory in Sciebo.
    The base folder contains product folders ("1", "2", ...) and a "Gocator" folder within each product.
    """
    # Load environment variables
    load_dotenv()

    # Retrieve credentials
    url = os.getenv("SCIEBO_URL")
    username = os.getenv("SCIEBO_USERNAME")
    password = os.getenv("SCIEBO_PASSWORD")

    try:
        # Initialize ownCloud client
        oc = owncloud.Client(url)
        oc.login(username, password)

        # Define the base folder containing product subfolders
        base_folder = "Experimentelle Daten Cu3dML/Produkte"
        all_features = []

        # List all product folders
        product_folders = [
            file for file in oc.list(base_folder)
            if file.file_type == "dir" and file.name.isdigit()  # Only include numbered product folders
        ]

        print(f"Found {len(product_folders)} product folder(s).")

        for product_folder in product_folders:
            product_path = product_folder.path
            product_id=os.path.split(os.path.dirname(product_path))[-1]
            gocator_folder_path = f"{product_path}/Gocator"

            try:
                # Check if the "Gocator" folder exists in the product folder
                gocator_files = [
                    file for file in oc.list(gocator_folder_path)
                    if file.file_type == "file" and file.name.endswith(".csv")
                ]

                if not gocator_files:
                    print(f"No Gocator CSV files found in {gocator_folder_path}. Skipping...")
                    continue

                print(f"Found {len(gocator_files)} Gocator CSV file(s) in {gocator_folder_path}.")

                for gocator_file in gocator_files:
                    remote_file_path = gocator_file.path

                    # Set output filename
                    output_filename = os.path.splitext(os.path.basename(remote_file_path))[0] + ".png"

                    try:
                        # Download the file
                        print(f"Processing Gocator file: {remote_file_path}")
                        remote_file_bytes = oc.get_file_contents(remote_file_path)

                        if not isinstance(remote_file_bytes, bytes):
                            raise TypeError(f"Expected bytes-like object but received: {type(remote_file_bytes)}")

                        # Preprocess the Gocator data
                        processed_data = DataHandler.preprocess_gocator(remote_file_bytes)

                        # Validate the processed data
                        assert isinstance(processed_data, pd.DataFrame), "Output should be a pandas DataFrame."
                        assert list(processed_data.columns) == ['X', 'Y', 'Z'], "DataFrame should have columns ['X', 'Y', 'Z']."
                        assert not processed_data.isnull().values.any(), "DataFrame should not contain NaN values."
                        assert len(processed_data) > 0, "Processed data should not be empty."

                        # Plot and save the Gocator point cloud
                        features = FeatureExtractor.extract_rectangular_deviation_features(
                            processed_data,
                            product_id=product_id
                        )
                        all_features.append(features)
                        print(all_features)

                    except Exception as e:
                        print(f"An error occurred while processing {remote_file_path}: {e}")
                
                if all_features:
                    features_df = pd.DataFrame(all_features)

                    # Sort the DataFrame by Product_ID after converting it to numeric
                    features_df = features_df.sort_values(by="Product_ID", key=lambda col: col.astype(int))

                    # Save the sorted DataFrame to a CSV file
                    output_csv_path = "gocator_features.csv"
                    features_df.to_csv(output_csv_path, index=False)
                    print(f"All features saved to {output_csv_path}")
                else:
                    print("No features were extracted.")

            except Exception as e:
                print(f"An error occurred while accessing {gocator_folder_path}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if oc:
            oc.logout()


if __name__ == "__main__":
    process_gocator_files_in_products()
