# scripts/upload_artifacts.py
import argparse
from pathlib import Path
from huggingface_hub import HfApi, HfFolder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_artifacts(repo_id: str, local_base_dir: str = "."):
    """
    Uploads models, adapters, and vocabularies to a Hugging Face Hub repository.
    """
    api = HfApi()
    local_path = Path(local_base_dir)

    logger.info(f"🚀 Starting upload to repository: {repo_id}")
    
    # Ensure the repository exists, create if it doesn't
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    logger.info(f"✅ Repository '{repo_id}' is ready.")

    # Define directories to upload and their target paths in the repo
    dirs_to_upload = {
        "models/production": "models/production",
        "models/adapters": "adapters",
        "vocabs": "vocabs"
    }

    for local_dir, repo_dir in dirs_to_upload.items():
        full_local_path = local_path / local_dir
        if full_local_path.exists() and full_local_path.is_dir():
            logger.info(f"\nUploading '{local_dir}' to '{repo_dir}' in the repo...")
            try:
                api.upload_folder(
                    folder_path=str(full_local_path),
                    path_in_repo=repo_dir,
                    repo_id=repo_id,
                    repo_type="model"
                )
                logger.info(f"✅ Successfully uploaded {local_dir}.")
            except Exception as e:
                logger.error(f"❌ Failed to upload {local_dir}. Error: {e}")
        else:
            logger.warning(f"⚠️  Skipping '{local_dir}': Directory not found at {full_local_path}")

    logger.info("\n🎉 Artifact upload process complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Universal Translation System artifacts to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="The Hugging Face Hub repository ID (e.g., 'your-username/your-repo-name').")
    args = parser.parse_args()
    
    if not HfFolder.get_token():
        logger.error("❌ Hugging Face token not found. Please log in first using 'huggingface-cli login'")
    else:
        upload_artifacts(args.repo_id)