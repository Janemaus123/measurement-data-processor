import os
import owncloud
import pandas as pd
from dotenv import load_dotenv
from io import StringIO
from measurement_processor.data_handler import DataHandler
from measurement_processor.visualizer import Visualizer
from measurement_processor.setup import FeatureExtractor

def process_timeseries_files_in_products():
    """
    Processes time-series CSV files from a structured directory in Sciebo.
    The base folder contains product folders ("1", "2", ...) and a "Timeseries" folder within each product.
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
            if file.file_type == "dir"  
        ]

        print(f"Found {len(product_folders)} product folder(s).")

        for product_folder in product_folders:
            product_path = product_folder.path
            product_id = os.path.split(os.path.dirname(product_path))[-1]
            timeseries_folder_path = f"{product_path}/Timeseries"

            try:
                # Check if the "Timeseries" folder exists in the product folder
                timeseries_files = [
                    file for file in oc.list(timeseries_folder_path)
                    if file.file_type == "file" and file.name.endswith(".csv")
                ]
                if not timeseries_files:
                    print(f"No Timeseries CSV files found in {timeseries_folder_path}. Skipping...")
                    continue
                
                print(f"Found {len(timeseries_files)} Timeseries CSV file(s) in {timeseries_folder_path}.")

                for timeseries_file in timeseries_files:
                    remote_file_path = timeseries_file.path

                    try:
                        # Download the file
                        print(f"Processing Timeseries file: {remote_file_path}")
                        remote_file_bytes = oc.get_file_contents(remote_file_path)
                        file_content = remote_file_bytes.decode("utf-8")
                        timeseries_df = pd.read_csv(StringIO(file_content), delimiter=';', decimal=',')
                        # Preprocess the time-series data   
                        features = ["Q_PG_N2"]
                        processed_data = DataHandler.preprocess_timeseries(timeseries_df, features=features)
                        Visualizer.plot_timeseries_from_array(processed_data, product_id=product_id, feature_names=features)
                        # Extract features from the processed time-series data
                        extracted_features = FeatureExtractor.extract_features_from_timeseries(
                            processed_data,
                            feature_columns=features
                        )
                        print(len(extracted_features))
                        # Append product ID and features to the results
                        for segment_features in extracted_features:
                            segment_features["Product_ID"] = product_id
                            all_features.append(segment_features)

                    except Exception as e:
                        print(f"An error occurred while processing {remote_file_path}: {e}")

            except Exception as e:
                print(f"An error occurred while accessing {timeseries_folder_path}: {e}")

        # Save all features to a CSV file
        if all_features:
            features_df = pd.DataFrame(all_features)

            # Sort by Product_ID
            features_df = features_df.sort_values(by="Product_ID", key=lambda col: col.astype(int))

            output_csv_path = "timeseries_features.csv"
            features_df.to_csv(output_csv_path, index=False)
            print(f"All features saved to {output_csv_path}")
        else:
            print("No features were extracted.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if oc:
            oc.logout()

if __name__ == "__main__":
    process_timeseries_files_in_products()
