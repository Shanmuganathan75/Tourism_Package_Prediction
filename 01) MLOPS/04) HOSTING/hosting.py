from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HGF_TOKEN"))
api.upload_folder(
    folder_path="01) MLOPS/03) DEPLOYMENT",                                    # the local folder containing your files
    repo_id="Shanmuganathan75/Tourism-Package-Prediction",                     # the target repo
    repo_type="space",                                                         # dataset, model, or space
    path_in_repo="",                                                            # optional: subfolder path inside the repo
)
