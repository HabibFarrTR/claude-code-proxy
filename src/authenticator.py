import json
import logging
import os

import requests
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials as OAuth2Credentials

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file
load_dotenv()


class AuthenticationError(Exception):
    """Custom exception for authentication failures."""

    pass


def get_gemini_credentials():
    """
    Authenticates with the custom endpoint and returns Vertex AI credentials.

    Returns:
        tuple: Contains project_id, location, and OAuth2Credentials object.
        Returns None, None, None on failure.

    Raises:
        AuthenticationError: If authentication fails for any reason.
    """
    workspace_id = os.getenv("WORKSPACE_ID")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-pro-preview-03-25")  # Use BIG_MODEL as default
    auth_url = os.getenv("AUTH_URL")

    if not all([workspace_id, auth_url]):
        raise AuthenticationError("Missing required environment variables: WORKSPACE_ID, AUTH_URL")

    logging.info(f"Authenticating with AI Platform for workspace {workspace_id} and model {model_name}")
    logging.info(
        "NOTE: Authentication requires AWS credentials to be set up via 'mltools-cli aws-login' before running"
    )

    payload = {
        "workspace_id": workspace_id,
        "model_name": model_name,
    }

    logging.info(f"Requesting temporary token from {auth_url} for workspace {workspace_id}")

    try:
        # Send the request with timeout
        resp = requests.post(auth_url, headers=None, json=payload, timeout=30)

        # Check status code before proceeding
        if resp.status_code != 200:
            error_msg = f"Authentication request failed with status {resp.status_code}"
            logging.error(f"{error_msg}: {resp.text}")
            raise AuthenticationError(error_msg)

        # Parse the response as JSON
        try:
            credentials_data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response: {e}"
            logging.error(f"{error_msg}. Response: {resp.text}")
            raise AuthenticationError(error_msg)

        # Check if token exists in response (matching README example)
        if "token" not in credentials_data:
            error_msg = credentials_data.get("message", "Token not found in response.")
            logging.error(f"Authentication failed: {error_msg}")
            raise AuthenticationError(f"Failed to retrieve token. Server response: {error_msg}")

        project_id = credentials_data.get("project_id")
        location = credentials_data.get("region")
        token = credentials_data["token"]
        expires_on_str = credentials_data.get("expires_on", "N/A")

        if not project_id or not location:
            raise AuthenticationError("Missing 'project_id' or 'region' in authentication response.")

        # Optional: Add check for token expiry if needed for long-running sessions,
        # but typically these tokens are short-lived and re-fetched per session.
        logging.info(f"Successfully fetched Gemini credentials. Valid until: {expires_on_str}")

        # Create credentials using the simplified constructor as in the README example
        temp_creds = OAuth2Credentials(token)
        return project_id, location, temp_creds

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error during authentication: {e}", exc_info=True)
        raise AuthenticationError(f"Failed to connect to authentication server: {e}. Check your network connection.")
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout during authentication: {e}", exc_info=True)
        raise AuthenticationError(f"Authentication request timed out: {e}. Server might be overloaded.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error during authentication: {e}", exc_info=True)
        # Include response text in the error message if available
        if "resp" in locals() and hasattr(resp, "text"):
            error_details = f"{e}. Response: {resp.text}"
        else:
            error_details = str(e)
        raise AuthenticationError(f"Network error connecting to auth endpoint: {error_details}")
    except json.JSONDecodeError as e:
        # Get the response text if 'resp' exists
        response_text = resp.text if "resp" in locals() and hasattr(resp, "text") else "No response available"
        error_msg = f"Failed to parse authentication response: {e}"
        logging.error(f"{error_msg}. Response text: {response_text}", exc_info=True)
        raise AuthenticationError(f"{error_msg}. Check server response format.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during authentication: {e}", exc_info=True)
        # Re-raise specific AuthenticationError or a generic one
        if isinstance(e, AuthenticationError):
            raise
        else:
            raise AuthenticationError(f"An unexpected error during authentication: {e}")
