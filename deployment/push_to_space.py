from huggingface_hub import login, create_repo, upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN", "PASTE_YOUR_HF_TOKEN_HERE")
SPACE_REPO = "subhaspace/mloperations"

login(HF_TOKEN)
create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

for filename in ["app.py", "requirements.txt", "Dockerfile"]:
    upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=SPACE_REPO,
        repo_type="space"
    )

print("Deployment files uploaded to Hugging Face Space.")
