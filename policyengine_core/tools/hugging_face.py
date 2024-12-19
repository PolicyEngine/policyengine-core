from huggingface_hub import hf_hub_download, login, HfApi
from getpass import getpass
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


def get_or_prompt_hf_token() -> str:
    """
    Either get the Hugging Face token from the environment,
    or prompt the user for it and store it in the environment.

    Returns:
        str: The Hugging Face token.
    """

    token = os.environ.get("HUGGING_FACE_TOKEN")
    if token is None:
        token = getpass(
            "Enter your Hugging Face token (or set HUGGING_FACE_TOKEN environment variable): "
        )
        # Optionally store in env for subsequent calls in same session
        os.environ["HUGGING_FACE_TOKEN"] = token

    return token
