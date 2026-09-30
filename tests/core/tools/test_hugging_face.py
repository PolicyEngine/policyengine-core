import itertools
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

    @pytest.mark.parametrize(
        "environ",
        [{}, {"HUGGING_FACE_TOKEN": ""}, {"HF_TOKEN": "hf_cached_token"}],
        ids=["token-unset", "token-empty", "hf-token-only"],
    )
    def test_download_private_repo_no_token(self, environ):
        """Private repo, no token, non-interactive: token=None, no prompt, no raise.

        With HUGGING_FACE_TOKEN unset (or empty, as when Dependabot runs CI
        without secrets), get_or_prompt_hf_token returns None instead of
        prompting, and download_huggingface_dataset passes token=None through
        to hf_hub_download rather than raising. huggingface_hub then falls
        back to its own cached token (HF_TOKEN or the `hf auth login` file)
        and raises its own error if that is missing too, so core must not
        fail early here. The third case sets only huggingface_hub's own
        HF_TOKEN: core still passes token=None, because resolving that token
        is huggingface_hub's job, not core's.

        The previous version of this test wrapped assert_not_called() inside
        pytest.raises(Exception). The download never raised, so control
        reached assert_not_called(), whose AssertionError (the download had
        been called) satisfied pytest.raises. The test passed precisely
        because the download was called, the opposite of what its name
        claimed to check.
        """
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"

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
                            "policyengine_core.tools.hugging_face.model_info"
                        ) as mock_model_info:
                            mock_response = MagicMock()
                            mock_response.status_code = 404
                            mock_response.headers = {}
                            mock_model_info.side_effect = RepositoryNotFoundError(
                                "Test error", response=mock_response
                            )

                            with pytest.warns(
                                UserWarning, match="no HUGGING_FACE_TOKEN"
                            ):
                                result = download_huggingface_dataset(
                                    test_repo, test_filename, test_version, test_dir
                                )

                            assert result is mock_download.return_value
                            mock_getpass.assert_not_called()
                            mock_download.assert_called_once_with(
                                repo_id=test_repo,
                                repo_type="model",
                                filename=test_filename,
                                revision=test_version,
                                token=None,
                                local_dir=test_dir,
                            )

    @pytest.mark.parametrize("gated", ["manual", "auto"])
    def test_download_gated_public_repo_passes_env_token(self, gated):
        """A public but gated repo must receive HUGGING_FACE_TOKEN.

        Regression test for PolicyEngine/policyengine-core#529: `private`
        is False for a gated repo, but the file download still needs a
        gate-approved token or the Hub answers 401.
        """
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"
        test_token = "gated_repo_test_token"

        with patch.dict(os.environ, {"HUGGING_FACE_TOKEN": test_token}, clear=True):
            with patch(
                "policyengine_core.tools.hugging_face.hf_hub_download"
            ) as mock_download:
                with patch(
                    "policyengine_core.tools.hugging_face.model_info"
                ) as mock_model_info:
                    mock_model_info.return_value = ModelInfo(
                        id=test_repo, private=False, gated=gated
                    )

                    download_huggingface_dataset(
                        test_repo, test_filename, test_version, test_dir
                    )

                    mock_download.assert_called_once_with(
                        repo_id=test_repo,
                        repo_type="model",
                        filename=test_filename,
                        revision=test_version,
                        token=test_token,
                        local_dir=test_dir,
                    )

    def test_download_private_flag_repo_passes_env_token(self):
        """A repo reported as private=True by model_info still gets the token."""
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"
        test_token = "private_repo_test_token"

        with patch.dict(os.environ, {"HUGGING_FACE_TOKEN": test_token}, clear=True):
            with patch(
                "policyengine_core.tools.hugging_face.hf_hub_download"
            ) as mock_download:
                with patch(
                    "policyengine_core.tools.hugging_face.model_info"
                ) as mock_model_info:
                    mock_model_info.return_value = ModelInfo(
                        id=test_repo, private=True, gated=False
                    )

                    download_huggingface_dataset(
                        test_repo, test_filename, test_version, test_dir
                    )

                    mock_download.assert_called_once_with(
                        repo_id=test_repo,
                        repo_type="model",
                        filename=test_filename,
                        revision=test_version,
                        token=test_token,
                        local_dir=test_dir,
                    )

    def test_download_gated_repo_non_interactive_without_token(self):
        """Gated repo in CI without secrets: no prompt, token=None is passed."""
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=False):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass"
                ) as mock_getpass:
                    with patch(
                        "policyengine_core.tools.hugging_face.hf_hub_download"
                    ) as mock_download:
                        with patch(
                            "policyengine_core.tools.hugging_face.model_info"
                        ) as mock_model_info:
                            mock_model_info.return_value = ModelInfo(
                                id=test_repo, private=False, gated="manual"
                            )

                            with pytest.warns(
                                UserWarning, match="no HUGGING_FACE_TOKEN"
                            ):
                                download_huggingface_dataset(
                                    test_repo, test_filename, test_version, test_dir
                                )

                            mock_getpass.assert_not_called()
                            mock_download.assert_called_once_with(
                                repo_id=test_repo,
                                repo_type="model",
                                filename=test_filename,
                                revision=test_version,
                                token=None,
                                local_dir=test_dir,
                            )

    @pytest.mark.parametrize("gated", [False, None])
    def test_download_public_ungated_repo_never_prompts(self, gated):
        """Public, ungated repo: no token lookup and no interactive prompt.

        Guards against widening the predicate so far that every public
        download on a developer machine asks for a token.
        """
        test_repo = "test_repo"
        test_filename = "test_filename"
        test_version = "test_version"
        test_dir = "test_dir"

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.isatty", return_value=True):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass"
                ) as mock_getpass:
                    # A string, so that a widened predicate fails on
                    # assert_not_called() below rather than on storing a
                    # MagicMock in os.environ.
                    mock_getpass.return_value = "prompted_token"
                    with patch(
                        "policyengine_core.tools.hugging_face.hf_hub_download"
                    ) as mock_download:
                        with patch(
                            "policyengine_core.tools.hugging_face.model_info"
                        ) as mock_model_info:
                            mock_model_info.return_value = ModelInfo(
                                id=test_repo, private=False, gated=gated
                            )

                            download_huggingface_dataset(
                                test_repo, test_filename, test_version, test_dir
                            )

                            mock_getpass.assert_not_called()
                            mock_download.assert_called_once_with(
                                repo_id=test_repo,
                                repo_type="model",
                                filename=test_filename,
                                revision=test_version,
                                token=None,
                                local_dir=test_dir,
                            )


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
    token=None, huggingface_hub falls back to its own cached token (for
    example HF_TOKEN or the `hf auth login` file) and raises its own 401 if
    that is missing too. The warning is what makes that 401 traceable to a
    missing HUGGING_FACE_TOKEN (#529).
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
        "gated": model_info answers with private=False, gated="manual", as
            policyengine/policyengine-uk-data-private does.
        "not-found": model_info raises RepositoryNotFoundError, which core
            treats as "probably private".
        """
        if lookup == "public":
            return {"return_value": ModelInfo(id="test_repo", private=False)}
        if lookup == "private-flag":
            return {"return_value": ModelInfo(id="test_repo", private=True)}
        if lookup == "gated":
            return {
                "return_value": ModelInfo(id="test_repo", private=False, gated="manual")
            }
        assert lookup == "not-found"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        return {
            "side_effect": RepositoryNotFoundError("Test error", response=mock_response)
        }

    @pytest.mark.parametrize("lookup", ["private-flag", "gated", "not-found"])
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

        # Exactly one warning, naming the repo, the fallback (including the
        # pre-0.34 login command), and both errors that can follow with
        # their 401 and 403 causes.
        assert len(record) == 1
        message = str(record[0].message)
        assert self.repo in message
        assert "HF_TOKEN" in message
        assert "hf auth login" in message
        assert "huggingface-cli login" in message
        assert "RepositoryNotFoundError" in message
        assert "GatedRepoError" in message
        assert "401" in message
        assert "403" in message
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
                "gated",
                {"HUGGING_FACE_TOKEN": "env_token"},
                False,
                None,
                "env_token",
                id="gated-env-token",
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


class TestTokenRoutingInvariants:
    """Exhaustive check of download_huggingface_dataset's token contract.

    Every combination of repo state, environment, TTY and prompt entry is run
    through the real function (with model_info, hf_hub_download and getpass
    mocked) and compared with the spec written out in _expected():

    - Only private, gated or not-found repos require authentication.
    - Such a repo gets HUGGING_FACE_TOKEN if it is non-empty; otherwise a
      prompt on a TTY; otherwise None. A public, ungated repo always gets
      None and never prompts.
    - The token passed on is None or a non-empty string, never "".
    - Exactly one no-token warning fires when a repo requiring
      authentication ends up with None, and none fires otherwise.
    """

    repo = "test_owner/test_repo"
    filename = "test_filename"

    REPO_STATES = {
        "ungated": dict(private=False, gated=False),
        "gated-none": dict(private=False, gated=None),
        "fields-missing": dict(),
        "gated-auto": dict(private=False, gated="auto"),
        "gated-manual": dict(private=False, gated="manual"),
        "private": dict(private=True, gated=False),
        "private-gated": dict(private=True, gated="manual"),
        "not-found": None,
    }
    ENVIRONS = {
        "token-unset": {},
        "token-empty": {"HUGGING_FACE_TOKEN": ""},
        "hf-token-only": {"HF_TOKEN": "hf_cached_token"},
        "token-set": {"HUGGING_FACE_TOKEN": "env_token"},
    }
    # itertools.product, not nested fors: only a comprehension's first
    # iterable can see class attributes.
    CASES = [
        pytest.param(state, environ, isatty, entry, id=f"{state}-{environ}-{tty}-{e}")
        for state, environ, (isatty, tty), (entry, e) in itertools.product(
            REPO_STATES,
            ENVIRONS,
            [(False, "no-tty"), (True, "tty")],
            [("", "empty-entry"), ("prompted_token", "entry")],
        )
    ]

    @staticmethod
    def _expected(state, environ, isatty, entry):
        requires_authentication = state in (
            "gated-auto",
            "gated-manual",
            "private",
            "private-gated",
            "not-found",
        )
        if not requires_authentication:
            return None, False, 0
        env_token = environ.get("HUGGING_FACE_TOKEN") or None
        if env_token is not None:
            return env_token, False, 0
        if isatty:
            token = entry or None
            return token, True, int(token is None)
        return None, False, 1

    @pytest.mark.parametrize(("state", "environ", "isatty", "entry"), CASES)
    def test_token_routing_matches_spec(self, state, environ, isatty, entry):
        fields = self.REPO_STATES[state]
        if fields is None:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.headers = {}
            model_info_config = {
                "side_effect": RepositoryNotFoundError(
                    "Test error", response=mock_response
                )
            }
        else:
            model_info_config = {"return_value": ModelInfo(id="test_repo", **fields)}
        env = self.ENVIRONS[environ]

        with patch.dict(os.environ, env, clear=True):
            with patch("os.isatty", return_value=isatty):
                with patch(
                    "policyengine_core.tools.hugging_face.getpass",
                    return_value=entry,
                ) as mock_getpass:
                    with patch(
                        "policyengine_core.tools.hugging_face.hf_hub_download"
                    ) as mock_download:
                        with patch(
                            "policyengine_core.tools.hugging_face.model_info",
                            **model_info_config,
                        ):
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                result = download_huggingface_dataset(
                                    self.repo, self.filename
                                )

        expected_token, expected_prompt, expected_warnings = self._expected(
            state, env, isatty, entry
        )
        assert result is mock_download.return_value
        mock_download.assert_called_once()
        token = mock_download.call_args.kwargs["token"]
        assert token == expected_token
        assert token is None or (isinstance(token, str) and token != "")
        assert mock_getpass.call_count == int(expected_prompt)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == expected_warnings
        assert all("no HUGGING_FACE_TOKEN" in str(w.message) for w in user_warnings)
