from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Create the directory if it doesn't exist
os.makedirs("/content/drive/MyDrive/01) GreatLearning/06) Advanced MLOPS/04) Tourism Package Prediction/01) MLOPS/02) MODEL BUILDING", exist_ok=True)

repo_id = "Shanmuganathan75/Tourism-Package-Prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="01) MLOPS/01) DATA/",
    repo_id=repo_id,
    repo_type=repo_type,
)
