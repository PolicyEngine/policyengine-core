from huggingface_hub import hf_hub_download, login, HfApi
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def download(repo: str, repo_filename: str, version: str = None):
    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )

    return hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=repo_filename,
        revision=version,
        token=token,
    )
