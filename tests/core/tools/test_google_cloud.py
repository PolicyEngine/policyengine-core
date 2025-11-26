import pytest
from policyengine_core.tools.google_cloud import parse_gs_url


class TestParseGsUrl:
    def test_basic_url(self):
        bucket, file_path, version = parse_gs_url("gs://my-bucket/file.h5")
        assert (bucket, file_path, version) == ("my-bucket", "file.h5", None)

    def test_subdirectory_url(self):
        bucket, file_path, version = parse_gs_url(
            "gs://my-bucket/data/2024/file.h5"
        )
        assert bucket == "my-bucket"
        assert file_path == "data/2024/file.h5"
        assert version is None

    def test_url_with_version(self):
        bucket, file_path, version = parse_gs_url(
            "gs://my-bucket/file.h5@12345"
        )
        assert (file_path, version) == ("file.h5", "12345")

    def test_subdirectory_with_version(self):
        bucket, file_path, version = parse_gs_url(
            "gs://my-bucket/path/to/file.h5@67890"
        )
        assert bucket == "my-bucket"
        assert (file_path, version) == ("path/to/file.h5", "67890")

    def test_deep_subdirectory(self):
        bucket, file_path, version = parse_gs_url(
            "gs://my-bucket/a/b/c/d/e/file.h5"
        )
        assert file_path == "a/b/c/d/e/file.h5"

    def test_invalid_url_no_gs_prefix(self):
        with pytest.raises(ValueError, match="Invalid gs:// URL format"):
            parse_gs_url("s3://my-bucket/file.h5")

    def test_invalid_url_no_file(self):
        with pytest.raises(ValueError, match="Invalid gs:// URL format"):
            parse_gs_url("gs://my-bucket")

    def test_invalid_url_no_file_with_slash(self):
        with pytest.raises(ValueError, match="Invalid gs:// URL format"):
            parse_gs_url("gs://my-bucket/")

    def test_bucket_with_dashes_and_dots(self):
        bucket, file_path, version = parse_gs_url(
            "gs://my-project.appspot.com/data/file.h5"
        )
        assert bucket == "my-project.appspot.com"
        assert file_path == "data/file.h5"

    def test_version_in_middle_of_path(self):
        # @ in subdirectory name should NOT be treated as version separator
        # Only the last @ should be used for version
        bucket, file_path, version = parse_gs_url(
            "gs://my-bucket/path@weird/file.h5@v1.0"
        )
        assert file_path == "path@weird/file.h5"
        assert version == "v1.0"
