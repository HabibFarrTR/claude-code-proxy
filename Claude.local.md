## Phase 1: Foundational Refactoring & Critical Tool Fixes

### Task 1: Refactor Authentication & SDK Initialization
- Goal: Improve performance and stability by initializing the Vertex AI SDK once at startup and managing credentials centrally.
- Actions:
  - Modify src/authenticator.py: Create a class or mechanism to fetch and refresh credentials periodically in the background (using
asyncio.create_task or a similar pattern). Store the current valid credentials (project, location, OAuth2Credentials) globally or in an
application state accessible by endpoints.
- Modify src/server.py:
      - Remove the per-request asyncio.to_thread(get_gemini_credentials) and asyncio.to_thread(vertexai.init) calls from create_message
and count_tokens.
  - Add startup event handler (@app.on_event("startup")) to perform the initial authentication and SDK initialization using the central
credential manager.
  - Endpoints should retrieve the current valid credentials from the central manager.
  - Investigate if GenerativeModel can be instantiated once or if it needs per-request instantiation using the refreshed credentials
(without global vertexai.init).
- Files: src/authenticator.py, src/server.py
- Rationale: Fixes performance bottleneck, potential race conditions, and aligns with standard SDK usage patterns.
- Verification: Server starts, basic requests succeed, logs show initialization happens once.

### Task 2: Correct Tool Schema Handling
- Goal: Fix the primary suspected cause of tool failures by aligning schema processing with Gemini documentation.
- Actions:
  - Modify src/converters.py: clean_gemini_schema:
      - Remove the aggressive cleaning logic.
  - Stop removing enum, default, and most format values.
  - Only perform minimal cleaning confirmed necessary (e.g., perhaps removing $schema if present, based on MCP example).
  - Preserve the original schema structure as much as possible.
  - Log warnings for any fields that are removed, explaining why.
- Files: src/converters.py
- Rationale: Ensures Gemini receives accurate tool definitions as required by its API. Directly addresses likely cause of
MALFORMED_FUNCTION_CALL.
- Verification: Test tool calls that previously failed, especially those using enum or specific formats. Observe if
MALFORMED_FUNCTION_CALL errors decrease. Examine logs to see the cleaned schema being sent.

### Task 3: Implement tool_config Mode Setting
- Goal: Correctly translate Anthropic's tool_choice to Gemini's tool_config modes (AUTO/ANY/NONE).
- Actions:
  - Modify src/server.py:
      - In create_message, determine the appropriate Gemini mode (AUTO, ANY, or NONE) based on the litellm_request_dict["tool_choice"].
  - Construct the tool_config dictionary (e.g., {"function_calling_config": {"mode": "ANY"}} or {"function_calling_config": {"mode":
"NONE"}}).
  - Pass this tool_config to the model.generate_content_async calls (uncomment and use the parameter).
- Files: src/server.py
- Rationale: Ensures Gemini respects the user's intent regarding whether function calling is allowed, forced, or disabled.
- Verification: Test requests with tool_choice: "any" and tool_choice: "none" to confirm Gemini's behavior matches (forcing a call or
refusing to call).

## Phase 2: Improving Robustness & Maintainability


### Task 4: Refactor Request ID Handling & Body Parsing
- Goal: Standardize request ID tracking and eliminate redundant request body parsing.
- Actions:
  - Modify src/server.py:
      - Ensure the middleware (log_requests_middleware) reliably generates and stores the request ID in request.state.request_id before
endpoint code runs.
  - Remove the separate request ID generation within create_message and count_tokens. Retrieve the ID solely from
request.state.request_id.
  - Ensure exception handlers also reliably retrieve the ID from request.state.
  - Remove the await raw_request.body() and subsequent JSON parsing from create_message and count_tokens.
- Modify src/models.py:
      - In the validate_and_map_model validator, store the original model name (v) into the info.data['original_model_name'] field
during validation.
- Files: src/server.py, src/models.py
- Rationale: Improves efficiency and makes request tracing through logs consistent and reliable.
- Verification: Logs show consistent request IDs across middleware, endpoints, and errors. Server performance potentially improves
slightly.

### Task 5: Enhance Error Handling & Retry Logic
- Goal: Make the server more resilient to transient upstream API errors and provide more accurate token counts.
- Actions:
  - Modify src/server.py:
      - Replace the basic while retries <= max_retries loop in create_message (non-streaming) with a more robust retry mechanism (e.g.,
using the tenacity library). Configure it to retry on specific transient google.api_core.exceptions (like ServiceUnavailable,
DeadlineExceeded) with exponential backoff.
  - In count_tokens, remove the inaccurate character-based fallback estimation. If the SDK's count_tokens_async fails, raise an
appropriate HTTPException (e.g., 502 or the upstream status code) instead of returning an estimate.
- Files: src/server.py
- Rationale: Improves robustness against temporary upstream issues and prevents misleading token count estimates.
- Verification: Simulate transient errors to confirm retry logic works. Confirm count_tokens fails correctly instead of estimating on
error.

### Task 6: Refine Converters (Focus on Tool History)
- Goal: Improve the reliability and maintainability of history conversion, especially for tool calls/results.
- Actions:
  - Modify src/converters.py: convert_anthropic_to_litellm: When processing an Anthropic tool_result, try to capture the function name
associated with the tool_use_id and store it in the intermediate tool role message.
- Modify src/converters.py: convert_litellm_messages_to_vertex_content:
      - Remove the backward search for the function name (Lines ~416-430). Retrieve the name directly from the intermediate tool message
prepared in the previous step.
  - Review the logic for constructing Part.from_function_response and Part.from_dict({"function_call": ...}) for correctness against SDK
examples.
  - Add more detailed logging within the conversion loops to trace how messages and parts are being transformed.
- Files: src/converters.py
- Rationale: Makes history conversion less brittle, easier to debug, and more aligned with how Gemini expects tool interactions in the
history.
- Verification: Test multi-turn tool conversations. Examine detailed logs to verify history structure sent to Gemini.

## Phase 3: Further Refinements


### Task 7: Refactor Streaming Converters
- Goal: Simplify the complex state management in streaming conversion.
- Actions:
  - Modify src/converters.py: convert_litellm_to_anthropic_sse and adapt_vertex_stream_to_litellm:
      - Consider introducing helper classes or state machines to manage the transitions between different chunk types (text, tool start,
tool args, tool end, etc.).
  - Break down the large functions into smaller, more focused helper functions.
  - Add extensive unit tests covering various chunk interleaving patterns and edge cases.
- Files: src/converters.py
- Rationale: Improves maintainability and reduces the likelihood of bugs in the complex streaming logic.
- Verification: Run existing streaming tests, add new ones for edge cases. Verify complex streaming scenarios (e.g., text followed by
tool, multiple tools) work correctly.

### Task 8: Review and Refine Model Handling
- Goal: Ensure model mapping and prefix handling are consistent and correct.
- Actions:
  - Modify src/models.py: map_model_name:
      - Clarify prefix handling. Decide if the function should return names with or without the aiplatform/ prefix, ensuring consistency
with src/config.py and SDK calls in src/server.py.
  - Make mapping logic more robust (e.g., prioritize exact matches over substring checks).
- Review src/config.py and src/server.py to ensure model names/prefixes are used consistently.
- Files: src/models.py, src/config.py, src/server.py
- Rationale: Prevents errors due to incorrect model identifiers being passed to the API.
- Verification: Test with different model name variations (Claude names, direct Gemini names with/without prefixes).

### Task 9: Add Comprehensive Tests
- Goal: Increase confidence in code changes and prevent regressions.
- Actions:
  - Review existing tests in tests/.
- Add new unit tests for:
      - src/converters.py functions (especially clean_gemini_schema, history conversion, streaming logic).
  - src/authenticator.py credential refreshing logic (if refactored).
  - src/models.py: map_model_name.
- Add new integration tests specifically targeting various tool use scenarios (single tool, multi-turn, different parameter types,
enums, forced choice, no choice).
- Files: tests/*
- Rationale: Essential for verifying fixes and ensuring future changes don't break functionality.
- Verification: All tests pass.
