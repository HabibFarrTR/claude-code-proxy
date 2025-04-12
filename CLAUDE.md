# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a specialized proxy server that allows Claude clients to interact with Thomson Reuters AIplatform's Vertex AI models. It provides a mapping from Anthropic Claude API to Vertex AI, enabling Claude Code clients to use AIplatform models without modification.

## Key Files
- `src/server.py`: The main server implementation, handling request/response formatting and model mapping
- `src/authenticator.py`: Handles authentication with Thomson Reuters AIplatform
- `tests/test_server.py`: Tests for AIplatform integration

## Build/Test Commands
- Install dependencies: `poetry install`
- Run server: `poetry run uvicorn src.server:app --host 0.0.0.0 --port 8082 --reload`
- Run tests (all): `poetry run pytest tests/`
- Run specific test: `poetry run pytest tests/test_server.py::test_aiplatform -v`
- Run with shell script: `./run_tests.sh` (for running with a test server instance)

## AIplatform Integration
- Authentication requires AWS credentials (run `mltools-cli aws-login` before starting)
- Uses OAuth2Credentials from a Thomson Reuters token
- Directly integrates with Vertex AI, bypassing LiteLLM
- Maps Claude model names to AIplatform models automatically

## Code Style Guidelines
- Imports: Group standard lib, third-party, and local imports, sorted alphabetically
- Formatting: Use 4-space indentation and follow PEP 8 guidelines
- Type annotations: Use Python's typing module for function parameters and return values
- Naming: Use snake_case for variables/functions and PascalCase for classes
- Error handling: Use try/except blocks with specific exceptions and meaningful error messages
- Logging: Use the built-in logging module with appropriate log levels
- Documentation: Use docstrings for functions, classes, and modules
- API design: Follow RESTful principles for endpoints with proper validation