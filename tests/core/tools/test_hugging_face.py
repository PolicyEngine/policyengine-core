import os
import pytest
from unittest.mock import patch
from policyengine_core.tools.hugging_face import get_or_prompt_hf_token


def test_get_token_from_environment():
    """Test retrieving token when it exists in environment variables"""
    test_token = "test_token_123"
    with patch.dict(
        os.environ, {"HUGGING_FACE_TOKEN": test_token}, clear=True
    ):
        result = get_or_prompt_hf_token()
        assert result == test_token


def test_get_token_from_user_input():
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


def test_empty_user_input():
    """Test handling of empty user input"""
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "policyengine_core.tools.hugging_face.getpass", return_value=""
        ):
            result = get_or_prompt_hf_token()
            assert result == ""
            assert os.environ.get("HUGGING_FACE_TOKEN") == ""


def test_environment_variable_persistence():
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
