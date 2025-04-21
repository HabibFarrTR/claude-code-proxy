# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a specialized proxy server that allows Claude clients to interact with Thomson Reuters AI Platform's Vertex AI models. It provides a mapping from Anthropic Claude API to Vertex AI, enabling Claude Code clients to use AI Platform models without modification.

## Key Files
- `src/server.py`: Main server with API endpoints and request handling
- `src/models.py`: Pydantic models for API requests/responses
- `src/utils.py`: Utility functions for logging and formatting
- `src/config.py`: Configuration settings and environment variables
- `src/authenticator.py`: Handles authentication with Thomson Reuters AI Platform
- `src/converters.py`: Format conversion between Anthropic and Vertex AI APIs
- `tests/test_server.py`: Tests for AI Platform integration

## Module Structure
- `server.py`: FastAPI application with endpoints for chat completions and token counting
- `models.py`: Pydantic data models for request/response validation and model mapping
- `utils.py`: Logging configuration, color formatting, and request visualization
- `config.py`: Environment variables and configuration constants
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
  - claude-3.5-haiku → gemini-2.0-flash
  - claude-3.7-sonnet → gemini-2.5-pro-preview-03-25

## Code Style Guidelines
- Imports: Group standard lib, third-party, and local imports, sorted alphabetically
- Formatting: Use 4-space indentation and follow PEP 8 guidelines
- Type annotations: Use Python's typing module for function parameters and return values
- Naming: Use snake_case for variables/functions and PascalCase for classes
- Error handling: Use try/except blocks with specific exceptions and meaningful error messages
- Logging: Use the built-in logging module with appropriate log levels
- Documentation: Use docstrings for modules, classes, and functions in Google style format
- Comments: Focus on explaining "why" not "what" and avoid trivial comments
- API design: Follow RESTful principles for endpoints with proper validation

## Known Limitations
- Tool Calling Differences: Gemini models don't support batch tool operations, unlike Claude.
- Function Call Formats: Some complex tool calls may require format adjustments.
- Complex Workflows: Multi-turn tool-using conversations may require special handling.
- System Instructions: There are differences in how system instructions are processed between APIs.
