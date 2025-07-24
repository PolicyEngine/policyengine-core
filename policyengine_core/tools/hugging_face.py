from huggingface_hub import (
    hf_hub_download,
    model_info,
    ModelInfo,
)
from huggingface_hub.errors import RepositoryNotFoundError
from getpass import getpass
import os
import warnings
import traceback

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def download_huggingface_dataset(
    repo: str,
    repo_filename: str,
    version: str = None,
    local_dir: str | None = None,
):
    """
    Download a dataset from the Hugging Face Hub.

    Args:
        repo (str): The Hugging Face repo name, in format "{org}/{repo}".
        repo_filename (str): The filename of the dataset.
        version (str, optional): The version of the dataset. Defaults to None.
        local_dir (str, optional): The local directory to save the dataset to. Defaults to None.
    """
    # Attempt connection to Hugging Face model_info endpoint
    # (https://huggingface.co/docs/huggingface_hub/v0.26.5/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)
    # Attempt to fetch model info to determine if repo is private
    # A RepositoryNotFoundError & 401 likely means the repo is private,
    # but this error will also surface for public repos with malformed URL, etc.
    try:
        fetched_model_info: ModelInfo = model_info(repo)
        is_repo_private: bool = fetched_model_info.private
    except RepositoryNotFoundError as e:
        # If this error type arises, it's likely the repo is private; see docs above
        is_repo_private = True
        pass
    except Exception as e:
        # Otherwise, there probably is just a download error
        raise Exception(
            f"Unable to download dataset {repo_filename} from Hugging Face. This may be because the repo "
            + f"is private, the URL is malformed, or the dataset does not exist. The full error is {traceback.format_exc()}"
        )

    authentication_token: str = None
    if is_repo_private:
        authentication_token: str = get_or_prompt_hf_token()

    return hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=repo_filename,
        revision=version,
        token=authentication_token,
        local_dir=local_dir,
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
