"""Authentication handler for Thomson Reuters AI Platform.

Provides authentication with the Thomson Reuters AI Platform to obtain
temporary credentials for accessing Vertex AI's Gemini models.
"""

import asyncio
import datetime
import json
import os
import time
from threading import Lock

import requests
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials

from src.config import GEMINI_BIG_MODEL
from src.utils import get_logger

logger = get_logger()


class AuthenticationError(Exception):
    """Custom exception for authentication failures with the Thomson Reuters API."""

    pass


class CredentialManager:
    """
    Centralized manager for Thomson Reuters AI Platform credentials.

    Handles credential acquisition, caching, and automatic refresh in the background.
    Provides thread-safe access to the current valid credentials.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one credential manager exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CredentialManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the credential manager."""
        if self._initialized:
            return

        self._credentials = None
        self._project_id = None
        self._location = None
        self._expiry_time = None
        self._refresh_task = None
        self._initialization_task = None
        self._initialized = True
        self._refresh_interval = 45 * 60  # Refresh 45 minutes before expiry (if known)
        self._default_expiry = 55 * 60  # Default expiry assumption: 55 minutes
        self._initialized_event = asyncio.Event()
        self._refresh_lock = asyncio.Lock()
        self._sdk_initialized = False

        logger.info("CredentialManager initialized (singleton instance)")

    async def initialize(self):
        """
        Initialize credentials on server startup.

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        if self._initialization_task is not None:
            # Initialization already in progress, wait for it
            logger.info("Waiting for existing credential initialization to complete...")
            await self._initialized_event.wait()
            return self._credentials is not None

        try:
            self._initialization_task = asyncio.create_task(self._initialize_impl())
            await self._initialization_task
            return True
        except Exception as e:
            logger.error(f"Failed to initialize credentials: {e}", exc_info=True)
            return False
        finally:
            self._initialization_task = None

    async def _initialize_impl(self):
        """Internal implementation of credential initialization."""
        logger.info("Starting initial credential acquisition...")
        try:
            await self._refresh_credentials()
            self._initialized_event.set()
            logger.info("Initial credential acquisition successful")
        except Exception as e:
            logger.error(f"Initial credential acquisition failed: {e}", exc_info=True)
            self._initialized_event.set()  # Set event even on failure so waiters don't hang
            raise

    async def get_credentials(self):
        """
        Get the current valid credentials, refreshing if necessary.

        Returns:
            tuple: (project_id, location, OAuth2Credentials)

        Raises:
            AuthenticationError: If credentials cannot be obtained
        """
        if not self._initialized_event.is_set():
            logger.info("Waiting for credential initialization to complete...")
            await self._initialized_event.wait()

        if self._credentials is None:
            logger.warning("No credentials available after initialization, attempting refresh")
            await self._refresh_credentials()
            if self._credentials is None:
                raise AuthenticationError("Failed to obtain credentials")

        # Check if credentials need refreshing (within 5 minutes of expiry)
        now = time.time()
        if self._expiry_time and now > (self._expiry_time - 300):
            logger.info("Credentials approaching expiry, triggering refresh")
            await self._refresh_credentials()

        return self._project_id, self._location, self._credentials

    async def _refresh_credentials(self):
        """
        Refresh the credentials from the Thomson Reuters auth endpoint.
        """
        async with self._refresh_lock:  # Prevent multiple concurrent refreshes
            try:
                logger.info("Refreshing credentials...")
                project_id, location, creds, expires_in = await asyncio.to_thread(get_gemini_credentials_with_expiry)

                # Update credentials atomically
                self._credentials = creds
                self._project_id = project_id
                self._location = location

                # Set expiry time
                if expires_in:
                    self._expiry_time = time.time() + expires_in
                    # Schedule refresh at 75% of the token lifetime
                    refresh_after = max(300, min(expires_in * 0.75, self._refresh_interval))
                else:
                    # Default expiry if not provided
                    self._expiry_time = time.time() + self._default_expiry
                    refresh_after = self._refresh_interval

                logger.info(
                    f"Credentials refreshed successfully. Scheduling next refresh in {refresh_after:.0f} seconds"
                )

                # Initialize Vertex SDK with the new credentials
                await self._initialize_vertex_sdk()

                # Schedule the next refresh
                if self._refresh_task:
                    self._refresh_task.cancel()

                self._refresh_task = asyncio.create_task(self._schedule_refresh(refresh_after))

            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}", exc_info=True)
                if self._credentials is None:
                    # Only propagate error if we don't have any credentials
                    raise

                # If we already have credentials, schedule a retry
                if self._refresh_task:
                    self._refresh_task.cancel()
                self._refresh_task = asyncio.create_task(self._schedule_refresh(300))  # Retry in 5 minutes

    async def _initialize_vertex_sdk(self):
        """Initialize the Vertex AI SDK with current credentials."""
        try:
            await asyncio.to_thread(
                vertexai.init, project=self._project_id, location=self._location, credentials=self._credentials
            )
            self._sdk_initialized = True
            logger.info(f"Vertex AI SDK initialized. Project: {self._project_id}, Location: {self._location}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI SDK: {e}", exc_info=True)
            self._sdk_initialized = False
            raise

    async def _schedule_refresh(self, delay):
        """Schedule a credential refresh after the specified delay."""
        try:
            await asyncio.sleep(delay)
            await self._refresh_credentials()
        except asyncio.CancelledError:
            # Task was cancelled, which is expected behavior
            pass
        except Exception as e:
            logger.error(f"Error in scheduled credential refresh: {e}", exc_info=True)
            # Schedule another retry
            self._refresh_task = asyncio.create_task(self._schedule_refresh(300))  # Retry in 5 minutes


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
    project_id, location, temp_creds, _ = get_gemini_credentials_with_expiry()
    return project_id, location, temp_creds


def get_gemini_credentials_with_expiry():
    """
    Similar to get_gemini_credentials but also extracts expiry information.

    Returns:
        tuple: Contains (project_id, location, OAuth2Credentials, expires_in_seconds)

    Raises:
        AuthenticationError: When authentication fails
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

        # Try to extract expiry time
        expires_in = None
        if expires_on_str and expires_on_str != "N/A":
            try:
                # Parse expiry time if in ISO format
                if "T" in expires_on_str:
                    expires_dt = datetime.datetime.fromisoformat(expires_on_str.replace("Z", "+00:00"))
                    now = datetime.datetime.now(expires_dt.tzinfo)
                    expires_in = (expires_dt - now).total_seconds()
                # Or if in Unix timestamp format
                elif expires_on_str.isdigit():
                    expires_in = int(expires_on_str) - time.time()
            except Exception as e:
                logger.warning(f"Could not parse expiry time '{expires_on_str}': {e}")

        if not all([token, project_id, location]):
            missing = [k for k in ["token", "project_id", "region"] if not credentials_data.get(k)]
            logger.error(f"Authentication failed: Missing {missing} in response. Data: {credentials_data}")
            raise AuthenticationError(f"Missing required fields ({missing}) in authentication response.")

        logger.info(
            f"Successfully fetched custom Gemini credentials. Project: {project_id}, Location: {location}, Valid until: {expires_on_str}"
        )
        if expires_in:
            logger.info(f"Token will expire in {expires_in:.0f} seconds")

        temp_creds = OAuth2Credentials(token)
        return project_id, location, temp_creds, expires_in

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
