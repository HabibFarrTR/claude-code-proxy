# Enhanced Claude Code Proxy Implementation

This document outlines how to deploy the enhanced Claude Code Proxy that connects Claude Code to Vertex AI with advanced tool support.

## Implementation Overview

We've enhanced the original Claude Code proxy with the following improvements:

1. **Enhanced Vertex AI Integration**: Using the beta client libraries for Vertex AI to support advanced features like tool usage and streaming.

2. **Better Tool Handling**: Proper conversion between Anthropic tool formats and Vertex AI function declarations.

3. **Improved Streaming**: Specialized streaming that handles both text content and tool usage in the Anthropic SSE format.

4. **Robust File Operations**: Enhanced pattern matching for different file operation patterns.

## Key Files

The enhanced implementation consists of several specialized files:

- `src/enhanced_api.py` - Enhanced Vertex AI client with tool support
- `src/enhanced_streaming.py` - Specialized streaming handler for Vertex AI responses
- Original files remain for backward compatibility

## Deployment Instructions

### Prerequisites

1. Install additional dependencies:
   ```bash
   pip install google-cloud-aiplatform protobuf
   ```

2. Set up authentication:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   export GOOGLE_CLOUD_PROJECT=your-project-id
   export GOOGLE_CLOUD_LOCATION=us-central1  # or your preferred region
   ```

### Configuration

Create a `.env` file with the following variables:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
USE_ENHANCED_API=true  # Set to true to use the enhanced implementation
```

### Deployment

Deploy using one of these methods:

1. **Direct deployment**:
   ```bash
   uvicorn src.server:app --host 0.0.0.0 --port 8082
   ```

2. **Docker deployment**:
   ```bash
   docker build -t claude-code-proxy .
   docker run -p 8082:8082 -e GOOGLE_CLOUD_PROJECT=your-project-id -e GOOGLE_CLOUD_LOCATION=us-central1 claude-code-proxy
   ```

## Usage with Claude Code

Configure Claude Code to use your proxy by setting the API endpoint in Claude Code settings:
```
https://your-proxy-domain:8082
```

## Implementation Notes

### Tool Format Conversion

The proxy converts between Anthropic's tool format and Vertex AI's function calling format:

**Anthropic Format**:
```json
{
  "tools": [
    {
      "name": "ReadFile",
      "description": "Read a file from the filesystem",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The path to the file"
          }
        },
        "required": ["path"]
      }
    }
  ]
}
```

**Vertex AI Format**:
```python
FunctionDeclaration(
    name="ReadFile",
    description="Read a file from the filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file"
            }
        },
        "required": ["path"]
    }
)
```

### Streaming Protocol

The enhanced streaming handler implements the Anthropic SSE format with these key events:

1. `message_start` - Starts a message
2. `content_block_start` - Starts a text or tool_use block
3. `content_block_delta` - Sends text content or tool input updates
4. `content_block_stop` - Ends a content block
5. `message_delta` - Provides stop reason and usage stats
6. `message_stop` - Ends the message
7. `[DONE]` - Signals end of stream

## Troubleshooting

- **Authentication Issues**: Verify your authentication credentials are properly set
- **Streaming Problems**: Check logs for errors in SSE formatting
- **Tool Usage Errors**: Verify tool schemas are compatible with Vertex AI

## Future Improvements

1. Implement token counting using Vertex AI's token counters
2. Add support for multi-modal content
3. Improve error handling and reporting