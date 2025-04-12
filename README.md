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
   *   `WORKSPACE_ID`: Your Thomson Reuters AIplatform workspace ID (REQUIRED).
   *   `AUTH_URL`: Authentication URL for Thomson Reuters AIplatform (default: "https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token").
   *   `MODEL_NAME`: The model name to use with AIplatform (default: "gemini-2.5-pro-preview-03-25").

   **IMPORTANT**: You must run `mltools-cli aws-login` before starting the server to set up AWS credentials.

4. **Run the server**:
   ```bash
   poetry run uvicorn src.server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

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

#### Easy Installation and Shell Integration

We provide two scripts to make using Claude Code with the proxy easier:

1. **Standalone Installer** - Installs the proxy without needing the full repo:
   ```bash
   # Get and run the installer
   curl -s https://raw.githubusercontent.com/YOUR-COMPANY/claude-code-proxy/main/scripts/install.sh | bash

   # Or specify installation directories
   ./scripts/install.sh --dir ~/apps/claude-proxy --bin ~/bin

   # After installation, you can use:
   claude-proxy start     # Start the proxy
   claudex --gemini       # Use Claude with Gemini proxy
   claudex                # Use Claude with native API
   ```

2. **Shell Profile Integration** - For switching between providers:
   ```bash
   # Add to your .bashrc or .zshrc
   source /path/to/claude-code-proxy/scripts/claude-profile.sh

   # Then you can use these commands:
   claude-orig            # Use Claude with native API
   claude-gem             # Use Claude with Gemini proxy
   proxy-start            # Start the proxy
   proxy-stop             # Stop the proxy
   proxy-status           # Check if proxy is running
   ```

## Model Mapping 🗺️

The proxy automatically maps Claude models to AIplatform models:

| Claude Model | AIplatform Mapping |
|--------------|-------------------|
| claude-3-haiku | aiplatform/gemini-2.0-flash |
| claude-3-sonnet | aiplatform/gemini-2.5-pro-preview-03-25 |
| claude-3-opus | aiplatform/gemini-2.5-pro-preview-03-25 |

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
- `claude-3-sonnet-20240229` becomes `aiplatform/gemini-2.5-pro-preview-03-25`
- `claude-3-haiku-20240307` becomes `aiplatform/gemini-2.0-flash`
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

| Module | Purpose |
|--------|---------|
| `server.py` | FastAPI endpoints and routing |
| `models.py` | Pydantic data models for validation |
| `api.py` | AIplatform client and request handling |
| `utils.py` | Helper functions and utilities |
| `converters.py` | Format conversion between APIs |
| `streaming.py` | Streaming response handling |
| `authenticator.py` | Thomson Reuters authentication |

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

### Installation Scripts 📦

For easy deployment and usage, we provide two helper scripts:

| Script | Purpose |
|--------|---------|
| scripts/install.sh | Standalone installer that downloads only required files and creates convenient executables |
| scripts/claude-profile.sh | Shell integration for easy switching between Claude API and Gemini proxy |

The installer script:
- Creates a virtual environment
- Installs only necessary dependencies
- Downloads required source files
- Creates command-line tools (`claude-proxy` and `claudex`)
- Allows you to specify installation directories

The shell profile:
- Provides convenient aliases for using Claude with different backends
- Allows starting/stopping the proxy server
- Can be integrated into your shell startup files (.bashrc/.zshrc)

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
