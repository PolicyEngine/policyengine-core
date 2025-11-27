import os
import tempfile
from getpass import getpass
from pathlib import Path


def parse_gs_url(url: str) -> tuple[str, str, str | None]:
    """
    Parse a Google Cloud Storage URL into components.

    Args:
        url: URL in format gs://bucket/path/to/file[@version]

    Returns:
        Tuple of (bucket, file_path, version)
        version is None if not specified
    """
    if not url.startswith("gs://"):
        raise ValueError(
            f"Invalid gs:// URL format: {url}. "
            "Expected format: gs://bucket/path/to/file[@version]"
        )

    # Remove the "gs://" prefix
    path = url[5:]
    parts = path.split("/", 1)

    if len(parts) < 2 or not parts[1]:
        raise ValueError(
            f"Invalid gs:// URL format: {url}. "
            "Expected format: gs://bucket/path/to/file[@version]"
        )

    bucket = parts[0]
    file_path = parts[1]

    version = None
    if "@" in file_path:
        file_path, version = file_path.rsplit("@", 1)

    return bucket, file_path, version


def download_gcs_file(
    bucket: str,
    file_path: str,
    version: str = None,
    local_path: str = None,
):
    """
    Download a file from Google Cloud Storage.

    Args:
        bucket: The GCS bucket name.
        file_path: The path to the file within the bucket.
        version: The generation/version of the file (optional).
        local_path: The local path to save the file to. If None, downloads to a temp directory.

    Returns:
        The local path where the file was saved.
    """
    try:
        from google.cloud import storage
        import google.auth
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required for gs:// URLs. "
            "Install it with: pip install google-cloud-storage"
        )

    credentials, project_id = _get_gcs_credentials()

    storage_client = storage.Client(
        credentials=credentials, project=project_id
    )

    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(file_path)

    if version:
        blob = bucket_obj.blob(file_path, generation=int(version))

    if local_path is None:
        # Download to a temp directory, preserving the filename
        filename = Path(file_path).name
        local_path = os.path.join(tempfile.gettempdir(), filename)

    blob.download_to_filename(local_path)
    return local_path


def upload_gcs_file(
    bucket: str,
    file_path: str,
    local_path: str,
    version_metadata: str = None,
):
    """
    Upload a file to Google Cloud Storage.

    Args:
        bucket: The GCS bucket name.
        file_path: The path to upload to within the bucket.
        local_path: The local path of the file to upload.
        version_metadata: Optional version string to store in blob metadata.
    """
    try:
        from google.cloud import storage
        import google.auth
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required for gs:// URLs. "
            "Install it with: pip install google-cloud-storage"
        )

    credentials, project_id = _get_gcs_credentials()

    storage_client = storage.Client(
        credentials=credentials, project=project_id
    )

    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(file_path)
    blob.upload_from_filename(local_path)

    if version_metadata:
        blob.metadata = {"version": version_metadata}
        blob.patch()

    return f"gs://{bucket}/{file_path}"


def _get_gcs_credentials():
    """
    Get GCS credentials, prompting for service account key path if needed.

    Returns:
        Tuple of (credentials, project_id)
    """
    try:
        import google.auth
        from google.auth import exceptions as auth_exceptions
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required for gs:// URLs. "
            "Install it with: pip install google-cloud-storage"
        )

    # First try default credentials (e.g., from gcloud auth, service account, etc.)
    try:
        credentials, project_id = google.auth.default()
        return credentials, project_id
    except auth_exceptions.DefaultCredentialsError:
        pass

    # If no default credentials, check for service account key in environment
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if key_path is None:
        key_path = getpass(
            "Enter path to GCS service account key JSON "
            "(or set GOOGLE_APPLICATION_CREDENTIALS): "
        )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    # Try again with the provided credentials
    credentials, project_id = google.auth.default()
    return credentials, project_id
