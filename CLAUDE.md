# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a specialized proxy server that allows Claude clients to interact with Thomson Reuters AI Platform's Vertex AI models. It provides a mapping from Anthropic Claude API to Vertex AI, enabling Claude Code clients to use AI Platform models without modification.

## Key Files
- `src/server.py`: Main server with API endpoints
- `src/models.py`: Pydantic models for API requests/responses
- `src/api.py`: AI Platform client for Vertex AI integration
- `src/utils.py`: Utility functions for model mapping and formatting
- `src/streaming.py`: Handler for streaming responses
- `src/authenticator.py`: Handles authentication with Thomson Reuters AI Platform
- `src/converters.py`: Format conversion between Anthropic and Vertex AI
- `tests/test_server.py`: Tests for AI Platform integration

## Module Structure
- `server.py`: API endpoints and FastAPI setup
- `models.py`: All Pydantic models for request/response validation
- `api.py`: AIplatformClient for direct Vertex AI integration and format conversion
- `utils.py`: Helper functions for model mapping, request processing, and logging
- `streaming.py`: Streaming response handler with Anthropic SSE compatibility
- `authenticator.py`: Thomson Reuters AI Platform authentication
- `converters.py`: Format conversion utilities for Anthropic/Vertex compatibility

## Build/Test Commands
- Install dependencies: `poetry install`
- Run server: `poetry run uvicorn src.server:app --host 0.0.0.0 --port 8082 --reload`
- Run tests (all): `poetry run pytest tests/`
- Run specific test: `poetry run pytest tests/test_server.py::test_aiplatform -v`
- Run with shell script [run_tests.sh](scripts/run_tests.sh) (for running with a test server instance)

## AI Platform Integration
- Authentication requires AWS credentials (run `mltools-cli aws-login` before starting)
- Uses OAuth2Credentials from a Thomson Reuters token
- Directly integrates with Vertex AI
- Maps Claude model names to AI Platform models automatically:
  - claude-3-haiku → gemini-2.0-flash
  - claude-3-sonnet/opus → gemini-2.5-pro-preview-03-25

## Code Style Guidelines
- Imports: Group standard lib, third-party, and local imports, sorted alphabetically
- Formatting: Use 4-space indentation and follow PEP 8 guidelines
- Type annotations: Use Python's typing module for function parameters and return values
- Naming: Use snake_case for variables/functions and PascalCase for classes
- Error handling: Use try/except blocks with specific exceptions and meaningful error messages
- Logging: Use the built-in logging module with appropriate log levels
- Documentation: Use docstrings for functions, classes, and modules
- API design: Follow RESTful principles for endpoints with proper validation
