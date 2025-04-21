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

## Testing Guidelines

1. **Test Isolation**: Unit tests should never depend on external services. Use mocks and fixtures.

2. **Test Coverage**: Aim for high test coverage, especially for critical components.

3. **Integration Test Marking**: Always mark tests that use external resources with `@pytest.mark.integration`.
   There is no need for a separate integration package. Note the integration tests involving LLMs are relatively
   cheap due to the caching layer, and are encouraged as it is hard to mock the LLM responses properly for many scenarios.

4. **Test Reliability**: Tests should be deterministic and not fail randomly.

### Example: Creating a New Test

```python
import pytest
from unittest.mock import MagicMock, patch

# Regular unit test
def test_feature():
    # Arrange
    input_data = "test"

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output

# Integration test with external dependency
@pytest.mark.integration
def test_external_feature():
    # This test will be skipped when running with -m "not integration"
    result = function_that_calls_api()
    assert result is not None
```

## Known Limitations 🚫

While this proxy enables using Claude Code with Thomson Reuters AI Platform's Gemini models, several significant limitations exist,
particularly impacting tool usage reliability:

### Tool Usage Limitations

1.  **Incorrect Schema Cleaning:** The `src/converters.py:clean_gemini_schema` function aggressively removes potentially valid and necessary
 schema information (e.g., `enum`, specific `format` types) required by the Gemini API. This is a **primary suspect** for causing
`MALFORMED_FUNCTION_CALL` errors and incorrect tool behavior. *(See Task 2)*
2.  **Missing `tool_config` Mapping:** Anthropic's `tool_choice` parameter (e.g., forcing a specific tool or `any` tool) is not correctly
translated into Gemini's required `tool_config` modes (`ANY`/`NONE`). The proxy likely defaults to `AUTO`, potentially ignoring the
requested tool choice behavior. *(See Task 3)*
3.  **Fragile History/Stream Conversion:** The logic in `src/converters.py` for converting tool calls, arguments, and results within the
conversation history and during streaming is complex and prone to errors. This can lead to corrupted state being sent to Gemini or malformed
 `tool_use` blocks being sent to the client. *(See Task 6 & 7)*
4.  **No Batch Tool Support (Gemini Limitation):** Gemini models do not support executing multiple tool calls in parallel within a single
turn, unlike Claude. The proxy does not serialize these, so client attempts to use batch operations (like `BatchTool`) will fail.

### Performance and Stability

1.  **Per-Request Auth/Initialization:** The server currently performs authentication and Vertex AI SDK initialization for every incoming
request (`src/server.py`). This adds significant latency and potential instability under load. *(See Task 1)*

### Other Issues

1.  **Minor Inconsistencies:** Issues like inconsistent request ID handling and potentially inaccurate model name mapping exist but are less
 critical than the tool usage problems. *(See Task 4 & 8)*
2. Add a New Section: "Current Development Focus"

## Current Development Focus

Efforts are underway to address the known limitations, particularly focusing on improving tool usage reliability and overall robustness. Key
 areas include:

*   Correcting tool schema handling (`clean_gemini_schema`).
*   Implementing Gemini's `tool_config` modes based on Anthropic's `tool_choice`.
*   Refactoring authentication/SDK initialization for performance and stability.
*   Improving the robustness of conversion logic (history/streaming) and error handling.

When working on the codebase, please prioritize changes aligning with these improvement goals (referenced by Task numbers in "Known
Limitations").
