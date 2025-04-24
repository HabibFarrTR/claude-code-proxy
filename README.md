# Anthropic API Proxy for Thomson Reuters AIplatform 🔄

**Use Anthropic clients (like Claude Code) with Thomson Reuters AIplatform backend.** 🤝

A specialized, modular proxy server that lets you use Anthropic clients with Thomson Reuters AIplatform (Vertex AI) models directly. 🌉

![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- Thomson Reuters AIplatform access
- AWS credentials configured via `mltools-cli aws-login`
- [Poetry](https://python-poetry.org/) installed

### Setup 🛠️

1. **Clone this repository**:

   ```bash
   git clone https://github.com/YOUR-COMPANY/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Install dependencies with Poetry**:

   ```bash
   poetry install
   ```

3. **Configure Environment Variables**:
   Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your AIplatform configuration:

   **AIplatform Configuration (Thomson Reuters):**

   - `WORKSPACE_ID`: Your Thomson Reuters AIplatform workspace ID (REQUIRED).
   - `AUTH_URL`: Authentication URL for Thomson Reuters AIplatform (default: "https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token").
   - `BIG_MODEL`: The high-capability model to use (default: "gemini-2.5-pro-preview-03-25").
   - `SMALL_MODEL`: The faster, lighter model to use (default: "gemini-2.0-flash").

   **IMPORTANT**: You must run `mltools-cli aws-login` before starting the server to set up AWS credentials.

4. **Run the server**:

   ```bash
   poetry run uvicorn src.server:app --host 0.0.0.0 --port 8082 --reload
   ```

   _(`--reload` is optional, for development)_

   Alternatively, use the poetry script:

   ```bash
   poetry run start src.server:app --host 0.0.0.0 --port 8082 --reload
   ```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):

   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:

   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use AIplatform models through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude models to AIplatform models:

| Claude Model      | AIplatform Mapping                      |
| ----------------- | --------------------------------------- |
| claude-3.5-haiku  | aiplatform/gemini-2.0-flash             |
| claude-3.7-sonnet | aiplatform/gemini-2.5-pro-preview-03-25 |

### Supported Models

#### AIplatform Models

The following AIplatform models are supported with automatic `aiplatform/` prefix handling:

- gemini-2.5-pro-preview-03-25 (high-capability model)
- gemini-2.0-flash (faster, lighter model)

### Model Prefix Handling

The proxy automatically adds the appropriate prefix to model names:

- AIplatform models get the `aiplatform/` prefix
- Claude models (haiku, sonnet, opus) are mapped to the appropriate AIplatform model

For example:

- `claude-3.7-sonnet` becomes `aiplatform/gemini-2.5-pro-preview-03-25`
- `claude-3.5-haiku` becomes `aiplatform/gemini-2.0-flash`
- Direct use: `aiplatform/gemini-2.5-pro-preview-03-25`

## How It Works 🧩

This specialized proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Authenticating** with Thomson Reuters AIplatform to get a Vertex AI token 🔑
3. **Mapping** Claude model names to the appropriate AIplatform models 🔄
4. **Directly calling** Vertex AI with the request 📤
5. **Converting** the response back to Anthropic format 🔄
6. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

### Architecture 🏗️

The proxy uses a modular architecture for maintainability and extensibility:

| Module             | Purpose                                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| `server.py`        | FastAPI application with endpoints for chat completions and token counting |
| `models.py`        | Pydantic data models for request/response validation and model mapping     |
| `utils.py`         | Logging configuration, color formatting, and request visualization         |
| `config.py`        | Environment variables and configuration constants                          |
| `authenticator.py` | Thomson Reuters AI Platform authentication                                 |
| `converters.py`    | Format conversion utilities between Anthropic and Vertex AI APIs           |

## Testing 🧪

The project includes tests for both streaming and non-streaming functionality, specifically for the AIplatform provider.

### Running Tests

We provide a convenient test script that handles server startup and cleanup automatically:

```bash
# Run all AI Platform tests
./scripts/run_tests.sh

# Run only specific test cases
./scripts/run_tests.sh -t test_aiplatform
./scripts/run_tests.sh -t test_aiplatform_with_tools
./scripts/run_tests.sh -t test_aiplatform_streaming

# Show full server logs
./scripts/run_tests.sh -v
```

IMPORTANT: Before running tests, make sure you've run `mltools-cli aws-login` first to set up your AWS credentials for Thomson Reuters AIplatform.

## Technical Details

### Authentication Flow

1. The server authenticates with Thomson Reuters AIplatform using AWS credentials
2. It receives a temporary token with project ID and region information
3. This token is converted to Google OAuth2Credentials
4. Vertex AI is initialized with these credentials
5. All API calls are made directly to Vertex AI using these credentials

### Message Format Conversion

1. Anthropic messages are converted to plain text for Vertex AI consumption
2. Content blocks (text, tool use, tool results) are flattened to text
3. Responses from Vertex AI are wrapped in Anthropic-compatible format
4. For streaming, the proxy simulates Anthropic's server-sent event structure

## Known Limitations 🚫

While this proxy enables using Claude Code with Thomson Reuters AIplatform's Gemini models, there are some inherent limitations:

### Tool Usage Limitations

1. **Batch Tool Operations**: Gemini models cannot process multiple tool calls simultaneously in the same way Claude does. The BatchTool feature of Claude Code will result in "MALFORMED_FUNCTION_CALL" errors.

2. **Function Call Format**: The underlying API formats for function/tool calling differ between Claude and Gemini. Complex tool use patterns that work with Claude may not translate properly to Gemini.

3. **Error Handling**: When function calls fail due to format differences, the proxy returns a generic "[Proxy Error: The model generated an invalid tool call and could not complete the request. Please try rephrasing.]" message.

### Model Behavior Differences

1. **Tool Output Processing**: Gemini and Claude may process and reason about tool outputs differently, affecting follow-up actions in agentic workflows.

2. **State Management**: Different approaches to maintaining chat state between models can affect multi-turn tool usage.

3. **Complex Workflows**: Advanced agentic workflows with multiple tool calls or batch operations will have degraded reliability compared to native Claude.

Despite state-of-the-art capabilities, Gemini models are not drop-in replacements for Claude in complex agentic workflows. Simple tool use scenarios work well, but more complex interactions may require adjustments to your workflow.
