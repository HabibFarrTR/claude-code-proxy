# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Install dependencies: `poetry install`
- Run server: `poetry run uvicorn server:app --host 0.0.0.0 --port 8082 --reload`
- Run server with script: `poetry run start server:app --host 0.0.0.0 --port 8082 --reload`
- Run all tests: `poetry run python tests.py`
- Run specific test types: `poetry run python tests.py --simple` or `--tools-only` or `--no-streaming`

## Code Style Guidelines
- Imports: Group standard lib, third-party, and local imports, sorted alphabetically
- Formatting: Use 4-space indentation and follow PEP 8 guidelines
- Type annotations: Use Python's typing module for function parameters and return values
- Naming: Use snake_case for variables/functions and PascalCase for classes
- Error handling: Use try/except blocks with specific exceptions and meaningful error messages
- Logging: Use the built-in logging module with appropriate log levels
- Documentation: Use docstrings for functions, classes, and modules
- API design: Follow RESTful principles for endpoints with proper validation