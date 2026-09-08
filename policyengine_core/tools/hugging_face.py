from huggingface_hub import (
    hf_hub_download,
    model_info,
    ModelInfo,
)
from huggingface_hub.errors import RepositoryNotFoundError
from getpass import getpass
import os
from pathlib import Path
import warnings
import traceback

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def parse_hf_url(url: str) -> tuple[str, str, str, str | None]:
    """
    Parse a Hugging Face URL into components.

    Args:
        url: URL in format hf://owner/repo/path/to/file[@version]

    Returns:
        Tuple of (owner, repo, file_path, version)
        version is None if not specified
    """
    parts = url.split("/")[2:]

    if len(parts) < 3:
        raise ValueError(
            f"Invalid hf:// URL format: {url}. "
            "Expected format: hf://owner/repo/path/to/file[@version]"
        )

    owner = parts[0]
    repo = parts[1]
    file_path = "/".join(parts[2:])

    version = None
    if "@" in file_path:
        file_path, version = file_path.rsplit("@", 1)

    return owner, repo, file_path, version


def download_huggingface_dataset(
    repo: str,
    repo_filename: str,
    version: str | None = None,
    local_dir: str | Path | None = None,
) -> str:
    """
    Download a PolicyEngine dataset file from the Hugging Face Hub.

    Args:
        repo: Hugging Face model repository identifier in ``owner/name``
            format.
        repo_filename: Path to the dataset file within the repository.
        version: Repository revision to download, such as a branch, tag, or
            commit hash. If omitted, Hugging Face uses the repository's
            default revision.
        local_dir: Directory in which to place the downloaded file. If
            omitted, Hugging Face uses its local cache.

    Returns:
        Path to the downloaded local file.
    """
    # Attempt connection to Hugging Face model_info endpoint
    # (https://huggingface.co/docs/huggingface_hub/v0.26.5/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)
    # Attempt to fetch model info to determine if repo is private
    # A RepositoryNotFoundError & 401 likely means the repo is private,
    # but this error will also surface for public repos with malformed URL, etc.
    try:
        fetched_model_info: ModelInfo = model_info(repo)
        is_repo_private = bool(fetched_model_info.private)
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

    authentication_token: str | None = None
    if is_repo_private:
        authentication_token = get_or_prompt_hf_token()

    return hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=repo_filename,
        revision=version,
        token=authentication_token,
        local_dir=local_dir,
    )


def get_or_prompt_hf_token() -> str | None:
    """
    Either get the Hugging Face token from the environment,
    or prompt the user for it and store it in the environment.

    Returns:
        str | None: The Hugging Face token, or None if not available
        and running non-interactively (e.g., in CI without secrets).
    """

    token = os.environ.get("HUGGING_FACE_TOKEN")
    # Treat empty string same as None (handles CI with missing secrets)
    if not token:
        # Check if running interactively before prompting
        if os.isatty(0):
            token = getpass(
                "Enter your Hugging Face token (or set HUGGING_FACE_TOKEN environment variable): "
            )
            # Store in env for subsequent calls in same session
            if token:
                os.environ["HUGGING_FACE_TOKEN"] = token
            else:
                # User entered empty string - return None
                return None
        else:
            # Non-interactive (CI) - return None instead of prompting
            return None

    return token
