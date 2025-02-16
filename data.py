# Install dependencies as needed:
# pip install kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile

# Set the file path (relative to the dataset root) for the video file
file_path = "/verified-data/data_btc_10s/barbell biceps curl/0b43a151-8995-4f7e-8568-45d65996a19c.mp4"

# Authenticate and download the video file using the Kaggle API
api = KaggleApi()
api.authenticate()
dataset = "philosopher0808/gym-workoutexercises-video"
print("Downloading video file from dataset...")
# This downloads the file as a zip archive and saves it to the current directory
api.dataset_download_file(dataset, file_path, path=".", force=True, quiet=False)

# Determine the downloaded zip file name (it appends ".zip" to the base file name)
downloaded_zip = os.path.join(".", os.path.basename(file_path)) + ".zip"

if os.path.exists(downloaded_zip):
    # Extract the zip file and remove the zip archive
    with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(downloaded_zip)
    print("Video downloaded and extracted successfully.")
else:
    downloaded_file = os.path.join(".", os.path.basename(file_path))
    if os.path.exists(downloaded_file):
        print("Video downloaded successfully as:", downloaded_file)
    else:
        print("Download failed or file not found.")