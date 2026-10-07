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

    Private repos and public-but-gated repos both require authentication,
    so for either the token is resolved by get_or_prompt_hf_token()
    (HUGGING_FACE_TOKEN, else an interactive prompt when stdin is a TTY,
    else None) and passed explicitly to hf_hub_download. Public, ungated
    repos are requested with token=None and never prompt; huggingface_hub
    may still apply its own implicitly configured token (HF_TOKEN or the
    login file) in that case.

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

    Warns:
        UserWarning: If the repo requires authentication but no
            HUGGING_FACE_TOKEN was available. The download still runs with
            token=None, so huggingface_hub applies its own cached token
            (HF_TOKEN, or the file written by `hf auth login`, which is
            `huggingface-cli login` before huggingface_hub 0.34) if it has
            one, unless HF_HUB_DISABLE_IMPLICIT_TOKEN is set; the warning
            explains a 401 that follows when there is no such token, or a
            403 when a gated repo has not approved it.
    """
    # Attempt connection to Hugging Face model_info endpoint
    # (https://huggingface.co/docs/huggingface_hub/v0.26.5/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)
    # Attempt to fetch model info to determine if the repo requires
    # authentication. Testing `private` alone is not enough: a public but
    # gated repo still answers model_info() without a token, but the file
    # download returns 401 unless a gate-approved token is sent.
    # ModelInfo.gated is False for ungated repos, "auto" or "manual" for
    # gated repos, and None if the field is absent, so a truthiness test
    # covers every state (both gate modes need an authenticated request).
    # A RepositoryNotFoundError & 401 likely means the repo is private,
    # but this error will also surface for public repos with malformed URL, etc.
    try:
        fetched_model_info: ModelInfo = model_info(repo)
        requires_authentication: bool = bool(fetched_model_info.private) or bool(
            fetched_model_info.gated
        )
    except RepositoryNotFoundError as e:
        # If this error type arises, it's likely the repo is private; see docs above
        requires_authentication = True
        pass
    except Exception as e:
        # Otherwise, there probably is just a download error
        raise Exception(
            f"Unable to download dataset {repo_filename} from Hugging Face. This may be because the repo "
            + f"is private, the URL is malformed, or the dataset does not exist. The full error is {traceback.format_exc()}"
        )

    authentication_token: str | None = None
    if requires_authentication:
        authentication_token = get_or_prompt_hf_token()
        if authentication_token is None:
            # Deliberately not an error: huggingface_hub resolves its own
            # cached token when token=None, so `hf auth login` users still
            # work. Warn so that the bare 401 huggingface_hub raises when
            # that fallback is empty too can be traced back here (#529).
            warnings.warn(
                f"Hugging Face repo '{repo}' requires authentication, but no "
                "HUGGING_FACE_TOKEN was available (the environment variable "
                "is unset or empty, and no token was entered at a prompt). "
                "huggingface_hub normally falls back to its own cached token "
                "if it has one (the HF_TOKEN environment variable, or the file "
                "written by `hf auth login`, which is `huggingface-cli login` "
                "before huggingface_hub 0.34). If the download that follows "
                "fails with RepositoryNotFoundError or GatedRepoError (a 401 "
                "when no token was sent, or a 403 when a gated repo has not "
                "approved the token), set HUGGING_FACE_TOKEN to a token whose "
                "account has access.",
                stacklevel=2,
            )

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
