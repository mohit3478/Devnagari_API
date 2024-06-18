from kaggle.api.kaggle_api_extended import KaggleApi
import os

def download_and_extract_dataset():
    # Set Kaggle credentials if necessary
    # from config.kaggle_config import set_kaggle_credentials
    # set_kaggle_credentials()

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Dataset path on Kaggle
    dataset_kaggle_path = 'ankushsachdeva27/handwritten-hindi-digits'

    # Directory where dataset will be downloaded
    dataset_download_path = './datasets/DevanagariHandwrittenCharacterDataset'

    # Ensure the directory exists or create it
    if not os.path.exists(dataset_download_path):
        os.makedirs(dataset_download_path)

    # Download and extract dataset
    api.dataset_download_files(dataset_kaggle_path, path=dataset_download_path, unzip=True)

if __name__ == "__main__":
    download_and_extract_dataset()
