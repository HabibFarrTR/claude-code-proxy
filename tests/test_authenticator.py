import os

import pytest
from dotenv import load_dotenv

from src.authenticator import AuthenticationError, get_gemini_credentials


def test_authenticator():
    # Load environment variables
    load_dotenv()

    # Verify environment variables are set
    required_vars = ["WORKSPACE_ID", "BIG_MODEL", "SMALL_MODEL", "AUTH_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        pytest.skip(f"Skipping test: Missing required environment variables: {', '.join(missing_vars)}")

    try:
        pid, loc, creds = get_gemini_credentials()
        assert pid is not None, "Project ID should not be None"
        assert loc is not None, "Location should not be None"
        assert creds is not None, "Credentials should not be None"
        print(f"Success! Project ID: {pid}, Location: {loc}, Credentials obtained.")
        # You could potentially test credentials validity here if needed
        # e.g., try making a simple Vertex AI call
    except AuthenticationError as e:
        pytest.fail(f"Authentication test failed: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")
