import os
import pytest
from unittest.mock import patch, MagicMock
from huggingface_hub import ModelInfo
from huggingface_hub.errors import RepositoryNotFoundError
from policyengine_core.tools.hugging_face import (
    get_or_prompt_hf_token,
    download_huggingface_dataset,
    parse_hf_url,
)


class TestHuggingFaceDownload:
    def test_download_public_repo(self):
        """Test downloading from a public repo"""
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"

        with patch(
            "policyengine_core.tools.hugging_face.hf_hub_download"
        ) as mock_download:
            with patch(
                "policyengine_core.tools.hugging_face.model_info"
            ) as mock_model_info:
                # Create mock ModelInfo object emulating public repo
                test_id = 0
                mock_model_info.return_value = ModelInfo(
                    id=test_id, private=False
                )

                download_huggingface_dataset(
                    test_repo, test_filename, test_version, test_dir
                )

                mock_download.assert_called_with(
                    repo_id=test_repo,
                    repo_type="model",
                    filename=test_filename,
                    revision=test_version,
                    local_dir=test_dir,
                    token=None,
                )

    def test_download_private_repo(self):
        """Test downloading from a private repo"""
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"

        with patch(
            "policyengine_core.tools.hugging_face.hf_hub_download"
        ) as mock_download:
            with patch(
                "policyengine_core.tools.hugging_face.model_info"
            ) as mock_model_info:
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_response.headers = {}
                mock_model_info.side_effect = RepositoryNotFoundError(
                    "Test error", response=mock_response
                )
                with patch(
                    "policyengine_core.tools.hugging_face.get_or_prompt_hf_token"
                ) as mock_token:
                    mock_token.return_value = "test_token"

                    download_huggingface_dataset(
                        test_repo, test_filename, test_version, test_dir
                    )
                    mock_download.assert_called_with(
                        repo_id=test_repo,
                        repo_type="model",
                        filename=test_filename,
                        revision=test_version,
                        token=mock_token.return_value,
                        local_dir=test_dir,
                    )

    def test_download_private_repo_no_token(self):
        """Test handling of private repo with no token"""
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"

        with patch(
            "policyengine_core.tools.hugging_face.hf_hub_download"
        ) as mock_download:
            with patch(
                "policyengine_core.tools.hugging_face.model_info"
            ) as mock_model_info:
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_response.headers = {}
                mock_model_info.side_effect = RepositoryNotFoundError(
                    "Test error", response=mock_response
                )
                with patch(
                    "policyengine_core.tools.hugging_face.get_or_prompt_hf_token"
                ) as mock_token:
                    mock_token.return_value = ""

                    with pytest.raises(Exception):
                        download_huggingface_dataset(
                            test_repo, test_filename, test_version, test_dir
                        )
                        mock_download.assert_not_called()


class TestGetOrPromptHfToken:
    def test_get_token_from_environment(self):
        """Test retrieving token when it exists in environment variables"""
        test_token = "test_token_123"
        with patch.dict(
            os.environ, {"HUGGING_FACE_TOKEN": test_token}, clear=True
        ):
            result = get_or_prompt_hf_token()
            assert result == test_token

    def test_get_token_from_user_input(self):
        """Test retrieving token via user input when not in environment"""
        test_token = "user_input_token_456"

        # Mock both empty environment and user input
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "policyengine_core.tools.hugging_face.getpass",
                return_value=test_token,
            ):
                result = get_or_prompt_hf_token()
                assert result == test_token

                # Verify token was stored in environment
                assert os.environ.get("HUGGING_FACE_TOKEN") == test_token

    def test_empty_user_input(self):
        """Test handling of empty user input"""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "policyengine_core.tools.hugging_face.getpass", return_value=""
            ):
                result = get_or_prompt_hf_token()
                assert result == ""
                assert os.environ.get("HUGGING_FACE_TOKEN") == ""

    def test_environment_variable_persistence(self):
        """Test that environment variable persists across multiple calls"""
        test_token = "persistence_test_token"

        # First call with no environment variable
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "policyengine_core.tools.hugging_face.getpass",
                return_value=test_token,
            ):
                first_result = get_or_prompt_hf_token()

            # Second call should use environment variable
            second_result = get_or_prompt_hf_token()

            assert first_result == second_result == test_token
            assert os.environ.get("HUGGING_FACE_TOKEN") == test_token


class TestParseHfUrl:
    def test_basic_url(self):
        owner, repo, file_path, version = parse_hf_url(
            "hf://owner/repo/file.h5"
        )
        assert (owner, repo, file_path, version) == (
            "owner",
            "repo",
            "file.h5",
            None,
        )

    def test_subdirectory_url(self):
        owner, repo, file_path, version = parse_hf_url(
            "hf://owner/repo/data/2024/file.h5"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert file_path == "data/2024/file.h5"
        assert version is None

    def test_url_with_version(self):
        owner, repo, file_path, version = parse_hf_url(
            "hf://owner/repo/file.h5@v1.0"
        )
        assert (file_path, version) == ("file.h5", "v1.0")

    def test_subdirectory_with_version(self):
        owner, repo, file_path, version = parse_hf_url(
            "hf://owner/repo/path/to/file.h5@v2.0"
        )
        assert (file_path, version) == ("path/to/file.h5", "v2.0")

    def test_deep_subdirectory(self):
        owner, repo, file_path, version = parse_hf_url(
            "hf://owner/repo/a/b/c/d/e/file.h5"
        )
        assert file_path == "a/b/c/d/e/file.h5"

    def test_invalid_url_too_short(self):
        with pytest.raises(ValueError, match="Invalid hf:// URL format"):
            parse_hf_url("hf://owner/repo")
