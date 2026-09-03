import os
import warnings
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
                mock_model_info.return_value = ModelInfo(id=test_id, private=False)

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
        with patch.dict(os.environ, {"HUGGING_FACE_TOKEN": test_token}, clear=True):
            result = get_or_prompt_hf_token()
            assert result == test_token

    def test_get_token_from_user_input(self):
        """Test retrieving token via user input when not in environment"""
        test_token = "user_input_token_456"

        # Mock empty environment, interactive mode, and user input
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=True):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass",
                    return_value=test_token,
                ):
                    result = get_or_prompt_hf_token()
                    assert result == test_token

                    # Verify token was stored in environment
                    assert os.environ.get("HUGGING_FACE_TOKEN") == test_token

    def test_empty_user_input(self):
        """Test handling of empty user input in interactive mode"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=True):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass",
                    return_value="",
                ):
                    result = get_or_prompt_hf_token()
                    # Empty input should return None (not stored)
                    assert result is None
                    # Empty token should not be stored
                    assert os.environ.get("HUGGING_FACE_TOKEN") is None

    def test_non_interactive_mode_returns_none(self):
        """Test that non-interactive mode (CI) returns None without prompting"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=False):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass"
                ) as mock_getpass:
                    result = get_or_prompt_hf_token()
                    assert result is None
                    # getpass should not be called in non-interactive mode
                    mock_getpass.assert_not_called()

    def test_empty_env_token_treated_as_none(self):
        """Test that empty string token in env is treated as missing"""
        with patch.dict(os.environ, {"HUGGING_FACE_TOKEN": ""}, clear=True):
            with patch("os.isatty", return_value=False):
                result = get_or_prompt_hf_token()
                # Empty string should be treated as None
                assert result is None

    def test_environment_variable_persistence(self):
        """Test that environment variable persists across multiple calls"""
        test_token = "persistence_test_token"

        # First call with no environment variable (interactive mode)
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=True):
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
        owner, repo, file_path, version = parse_hf_url("hf://owner/repo/file.h5")
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
        owner, repo, file_path, version = parse_hf_url("hf://owner/repo/file.h5@v1.0")
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


class TestNoTokenWarning:
    """download_huggingface_dataset warns when it passes token=None for a
    repo that needs authentication.

    Core deliberately does not raise or prompt in that case (#422): with
    token=None, huggingface_hub falls back to its own cached token (HF_TOKEN
    or the `hf auth login` file) and raises its own 401 if that is missing
    too. The warning is what makes that 401 traceable to a missing or
    unapproved HUGGING_FACE_TOKEN (#529).
    """

    repo = "test_owner/test_repo"
    filename = "test_filename"
    version = "test_version"
    local_dir = "test_dir"

    def _download(self):
        return download_huggingface_dataset(
            self.repo, self.filename, self.version, self.local_dir
        )

    def _assert_downloaded_with(self, mock_download, token):
        mock_download.assert_called_once_with(
            repo_id=self.repo,
            repo_type="model",
            filename=self.filename,
            revision=self.version,
            token=token,
            local_dir=self.local_dir,
        )

    @staticmethod
    def _lookup_response(lookup):
        """Configure model_info for the given repo visibility.

        "public": the repo is public, so no token is ever needed.
        "private-flag": model_info answers with private=True.
        "not-found": model_info raises RepositoryNotFoundError, which core
            treats as "probably private".
        """
        if lookup == "public":
            return {"return_value": ModelInfo(id="test_repo", private=False)}
        if lookup == "private-flag":
            return {"return_value": ModelInfo(id="test_repo", private=True)}
        assert lookup == "not-found"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        return {
            "side_effect": RepositoryNotFoundError("Test error", response=mock_response)
        }

    @pytest.mark.parametrize("lookup", ["private-flag", "not-found"])
    @pytest.mark.parametrize(
        "environ",
        [{}, {"HUGGING_FACE_TOKEN": ""}, {"HF_TOKEN": "hf_cached_token"}],
        ids=["token-unset", "token-empty", "hf-token-only"],
    )
    def test_warns_when_no_token_resolved_non_interactively(self, lookup, environ):
        """No HUGGING_FACE_TOKEN, no TTY: warn, then pass token=None through.

        The hf-token-only case pins that the warning still fires when only
        huggingface_hub's own HF_TOKEN is set: core resolved nothing, and
        the warning itself says the fallback will be used if present.
        """
        model_info_config = self._lookup_response(lookup)

        with patch.dict(os.environ, environ, clear=True):
            with patch("os.isatty", return_value=False):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass"
                ) as mock_getpass:
                    mock_getpass.return_value = "prompted_token"
                    with patch(
                        "policyengine_core.tools.hugging_face.hf_hub_download"
                    ) as mock_download:
                        with patch(
                            "policyengine_core.tools.hugging_face.model_info",
                            **model_info_config,
                        ):
                            with pytest.warns(
                                UserWarning, match="no HUGGING_FACE_TOKEN"
                            ) as record:
                                result = self._download()

        # Behaviour is unchanged: no prompt, no raise, token=None passed on.
        assert result is mock_download.return_value
        mock_getpass.assert_not_called()
        self._assert_downloaded_with(mock_download, token=None)

        # Exactly one warning, naming the repo, the fallback, and the 401.
        assert len(record) == 1
        message = str(record[0].message)
        assert self.repo in message
        assert "HF_TOKEN" in message
        assert "hf auth login" in message
        assert "401" in message
        # stacklevel=2: the warning points at the caller, not at core.
        assert record[0].filename == __file__

    def test_warns_when_interactive_prompt_left_empty(self):
        """TTY present but the user enters nothing: same warning, token=None."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=True):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass",
                    return_value="",
                ) as mock_getpass:
                    with patch(
                        "policyengine_core.tools.hugging_face.hf_hub_download"
                    ) as mock_download:
                        with patch(
                            "policyengine_core.tools.hugging_face.model_info",
                            **self._lookup_response("private-flag"),
                        ):
                            with pytest.warns(
                                UserWarning, match="no HUGGING_FACE_TOKEN"
                            ):
                                self._download()

        mock_getpass.assert_called_once()
        self._assert_downloaded_with(mock_download, token=None)

    @pytest.mark.parametrize(
        ("lookup", "environ", "isatty", "prompted", "expected_token"),
        [
            pytest.param("public", {}, False, None, None, id="public-repo"),
            pytest.param(
                "public",
                {"HUGGING_FACE_TOKEN": "env_token"},
                False,
                None,
                None,
                id="public-repo-ignores-env-token",
            ),
            pytest.param(
                "private-flag",
                {"HUGGING_FACE_TOKEN": "env_token"},
                False,
                None,
                "env_token",
                id="private-flag-env-token",
            ),
            pytest.param(
                "not-found",
                {"HUGGING_FACE_TOKEN": "env_token"},
                False,
                None,
                "env_token",
                id="not-found-env-token",
            ),
            pytest.param(
                "private-flag",
                {},
                True,
                "prompted_token",
                "prompted_token",
                id="private-flag-prompted-token",
            ),
        ],
    )
    def test_no_warning_when_a_token_is_passed_or_not_needed(
        self, lookup, environ, isatty, prompted, expected_token
    ):
        """Public repos pass token=None without warning; a resolved token
        never warns. Guards against warning on every public download."""
        with patch.dict(os.environ, environ, clear=True):
            with patch("os.isatty", return_value=isatty):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass",
                    return_value=prompted,
                ):
                    with patch(
                        "policyengine_core.tools.hugging_face.hf_hub_download"
                    ) as mock_download:
                        with patch(
                            "policyengine_core.tools.hugging_face.model_info",
                            **self._lookup_response(lookup),
                        ):
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", UserWarning)
                                self._download()

        self._assert_downloaded_with(mock_download, token=expected_token)
