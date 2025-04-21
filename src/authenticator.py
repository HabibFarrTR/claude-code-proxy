"""Authentication handler for Thomson Reuters AI Platform.

Provides authentication with the Thomson Reuters AI Platform to obtain
temporary credentials for accessing Vertex AI's Gemini models.
"""

import json
import os

import requests
from google.oauth2.credentials import Credentials as OAuth2Credentials

from src.config import GEMINI_BIG_MODEL
from src.utils import get_logger

logger = get_logger()


class AuthenticationError(Exception):
    """Custom exception for authentication failures with the Thomson Reuters API."""

    pass


def get_gemini_credentials():
    """
    Authenticate with the Thomson Reuters endpoint and retrieve Vertex AI credentials.

    Contacts the Thomson Reuters authentication service to obtain temporary
    credentials for accessing Vertex AI's Gemini models. The credentials are
    workspace-specific and model-specific.

    Returns:
        tuple: Contains (project_id, location, OAuth2Credentials) for Vertex AI

    Raises:
        AuthenticationError: When authentication fails due to missing environment
            variables, network issues, or invalid responses
    """
    workspace_id = os.getenv("WORKSPACE_ID")
    model_name_for_auth = os.getenv("GEMINI_BIG_MODEL", GEMINI_BIG_MODEL)
    auth_url = os.getenv("AUTH_URL")

    if not all([workspace_id, auth_url]):
        logger.error("Missing required environment variables: WORKSPACE_ID and/or AUTH_URL")
        raise AuthenticationError("Missing required environment variables: WORKSPACE_ID, AUTH_URL")

    logger.info(
        f"Attempting custom authentication for workspace '{workspace_id}' (using model '{model_name_for_auth}' for auth)"
    )
    logger.debug(f"Authentication URL: {auth_url}")

    payload = {"workspace_id": workspace_id, "model_name": model_name_for_auth}
    logger.info(f"Requesting temporary token from {auth_url}")

    try:
        resp = requests.post(auth_url, headers=None, json=payload, timeout=30)
        if resp.status_code != 200:
            error_msg = f"Authentication request failed: Status {resp.status_code}"
            logger.error(f"{error_msg}. Response: {resp.text[:500]}...")
            raise AuthenticationError(f"{error_msg}. Check auth server and credentials.")

        try:
            credentials_data = resp.json()
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from auth server: {e}"
            logger.error(f"{error_msg}. Raw Response: {resp.text[:500]}...")
            raise AuthenticationError(error_msg)

        token = credentials_data.get("token")
        project_id = credentials_data.get("project_id")
        location = credentials_data.get("region")
        expires_on_str = credentials_data.get("expires_on", "N/A")

        if not all([token, project_id, location]):
            missing = [k for k in ["token", "project_id", "region"] if not credentials_data.get(k)]
            logger.error(f"Authentication failed: Missing {missing} in response. Data: {credentials_data}")
            raise AuthenticationError(f"Missing required fields ({missing}) in authentication response.")

        logger.info(
            f"Successfully fetched custom Gemini credentials. Project: {project_id}, Location: {location}, Valid until: {expires_on_str}"
        )
        temp_creds = OAuth2Credentials(token)  # This is the object LiteLLM choked on
        return project_id, location, temp_creds

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during authentication to {auth_url}: {e}", exc_info=True)
        raise AuthenticationError(f"Failed to connect to authentication server: {e}. Check network and URL.")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during authentication to {auth_url}: {e}", exc_info=True)
        raise AuthenticationError(f"Authentication request timed out: {e}. Auth server might be down or overloaded.")
    except requests.exceptions.RequestException as e:
        response_text = getattr(e.response, "text", "(No response text available)")
        logger.error(
            f"Network error during authentication to {auth_url}: {e}. Response: {response_text}", exc_info=True
        )
        raise AuthenticationError(f"Network error connecting to auth endpoint: {e}")
    except AuthenticationError:
        raise  # Re-raise specific auth errors
    except Exception as e:
        logger.error(f"Unexpected error during custom authentication: {e}", exc_info=True)
        raise AuthenticationError(f"Unexpected error during authentication: {e}")
