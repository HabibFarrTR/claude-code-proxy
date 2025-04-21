import json
import logging
from typing import Any, Dict, List

import pytest
import vertexai
from fastapi.testclient import TestClient

from src.converters import clean_gemini_schema
from src.models import (
    ContentBlockText,
    Message,
    MessagesRequest,
    ToolDefinition,
    ToolInputSchema,
)
from src.server import app, credential_manager

# Configure logging for more detailed output during tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a test client
client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
async def setup_authentication():
    """Initialize authentication and Vertex AI SDK once before running tests"""
    # Get credentials from the credential manager
    try:
        # This is similar to what the server does at startup
        await credential_manager.initialize()
        project_id, location, credentials = await credential_manager.get_credentials()
        logger.info(f"Test authentication successful. Project: {project_id}, Location: {location}")
        # Initialize the vertexai SDK
        vertexai.init(project=project_id, location=location, credentials=credentials)
        logger.info("Vertex AI SDK initialized for tests")
    except Exception as e:
        logger.error(f"Authentication failed in test setup: {e}")
        pytest.skip("Authentication failed, skipping tests")


def create_tool_request(tools: List[Dict], message_content: str = "Use the tool") -> Dict:
    """Create a request payload for testing tools"""
    tool_definitions = []
    for tool in tools:
        if isinstance(tool, dict):
            input_schema = ToolInputSchema.model_validate(tool["parameters"])
            tool_def = ToolDefinition(
                name=tool["name"], description=tool.get("description", ""), input_schema=input_schema
            )
            tool_definitions.append(tool_def)

    request = MessagesRequest(
        model="claude-3.7-sonnet",
        max_tokens=1000,
        messages=[Message(role="user", content=[ContentBlockText(type="text", text=message_content)])],
        tools=tool_definitions,
        temperature=0.5,
    )
    return request.model_dump(by_alias=True)


def clean_schema_test(schema: Dict) -> Dict:
    """Run a schema through clean_gemini_schema and log before/after for debugging"""
    logger.info(f"Original schema:\n{json.dumps(schema, indent=2)}")
    cleaned = clean_gemini_schema(schema)
    logger.info(f"Cleaned schema:\n{json.dumps(cleaned, indent=2)}")
    return cleaned


@pytest.mark.integration
def test_glob_tool_basic():
    """Test basic GlobTool functionality"""
    glob_tool = {
        "name": "GlobTool",
        "description": "Fast file pattern matching tool",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                "path": {"type": "string", "description": "The directory to search in"},
            },
            "required": ["pattern"],
        },
    }

    # Just test the cleanup to ensure the schema is as expected
    clean_schema_test(glob_tool["parameters"])

    request_data = create_tool_request([glob_tool], "List all Python files in the current directory using GlobTool.")

    # Send the request to our API
    response = client.post("/v1/messages", json=request_data)

    logger.info(f"Response status: {response.status_code}")
    logger.info(f"Response body: {json.dumps(response.json(), indent=2)}")

    # Check if the response is successful
    assert response.status_code == 200

    # Check if the tool was used in the response
    response_data = response.json()
    assert "content" in response_data
    # Check for a tool use block in the content
    content_text = response_data.get("content", [{}])[0].get("text", "")
    assert "GlobTool" in content_text


@pytest.mark.integration
def test_batch_tool_basic():
    """Test basic BatchTool functionality"""
    batch_tool = {
        "name": "BatchTool",
        "description": "Batch execution tool that runs multiple tools",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "A short description of the batch operation"},
                "invocations": {
                    "type": "array",
                    "description": "The list of tool invocations to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "The name of the tool to invoke"},
                            "input": {
                                "type": "object",
                                "description": "The input to pass to the tool",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["tool_name", "input"],
                    },
                },
            },
            "required": ["description", "invocations"],
        },
    }

    glob_tool = {
        "name": "GlobTool",
        "description": "Fast file pattern matching tool",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                "path": {"type": "string", "description": "The directory to search in"},
            },
            "required": ["pattern"],
        },
    }

    # Check cleaned schemas
    clean_schema_test(batch_tool["parameters"])
    clean_schema_test(glob_tool["parameters"])

    request_data = create_tool_request(
        [batch_tool, glob_tool],
        "Use BatchTool to run two GlobTool operations: 1) Find all Python files and 2) Find all Markdown files",
    )

    # Send the request to our API
    response = client.post("/v1/messages", json=request_data)

    logger.info(f"Response status: {response.status_code}")
    logger.info(f"Response body: {json.dumps(response.json(), indent=2)}")

    # Check if the response is successful
    assert response.status_code == 200

    # Check if BatchTool was used in the response
    response_data = response.json()
    assert "content" in response_data
    content_text = response_data.get("content", [{}])[0].get("text", "")
    assert "BatchTool" in content_text


@pytest.mark.integration
def test_multiple_nested_tools():
    """Test multiple tools including nested schemas with additionalProperties"""
    batch_tool = {
        "name": "BatchTool",
        "description": "Batch execution tool that runs multiple tools",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "A short description of the batch operation"},
                "invocations": {
                    "type": "array",
                    "description": "The list of tool invocations to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "The name of the tool to invoke"},
                            "input": {
                                "type": "object",
                                "description": "The input to pass to the tool",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["tool_name", "input"],
                    },
                },
            },
            "required": ["description", "invocations"],
        },
    }

    glob_tool = {
        "name": "GlobTool",
        "description": "Fast file pattern matching tool",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                "path": {"type": "string", "description": "The directory to search in"},
            },
            "required": ["pattern"],
        },
    }

    grep_tool = {
        "name": "GrepTool",
        "description": "Fast content search tool",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The pattern to search for"},
                "path": {"type": "string", "description": "The directory to search in"},
                "include": {"type": "string", "description": "File pattern to include"},
            },
            "required": ["pattern"],
        },
    }

    view_tool = {
        "name": "View",
        "description": "File reading tool",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The file path to read"},
                "offset": {"type": "number", "description": "Line offset"},
                "limit": {"type": "number", "description": "Line limit"},
            },
            "required": ["file_path"],
        },
    }

    # Log all the cleaned schemas for debugging
    for tool in [batch_tool, glob_tool, grep_tool, view_tool]:
        logger.info(f"Cleaning schema for {tool['name']}")
        clean_schema_test(tool["parameters"])

    prompt = """
    I'd like you to help me analyze this codebase. Please perform these steps:
    1. Use GlobTool to find all Python files
    2. Use GrepTool to find files containing "clean_gemini_schema"
    3. Use BatchTool to run both the View tool on server.py and converters.py at the same time
    """

    request_data = create_tool_request([batch_tool, glob_tool, grep_tool, view_tool], prompt)

    # Send the request to our API
    response = client.post("/v1/messages", json=request_data)

    logger.info(f"Response status: {response.status_code}")
    if response.status_code != 200:
        logger.error(f"Error response: {response.text}")
    else:
        logger.info(f"Response body: {json.dumps(response.json(), indent=2)}")

    # Check if the response is successful
    assert response.status_code == 200


@pytest.mark.integration
def test_tool_with_enum_values():
    """Test tool with enum values to ensure they're preserved"""
    edit_cell_tool = {
        "name": "NotebookEditCell",
        "description": "Edits a cell in a Jupyter notebook",
        "parameters": {
            "type": "object",
            "properties": {
                "notebook_path": {"type": "string", "description": "Path to the notebook"},
                "cell_number": {"type": "number", "description": "Cell index"},
                "new_source": {"type": "string", "description": "New cell content"},
                "cell_type": {"type": "string", "enum": ["code", "markdown"], "description": "Cell type"},
                "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"], "description": "Edit mode"},
            },
            "required": ["notebook_path", "cell_number", "new_source"],
        },
    }

    # Log cleaned schema
    clean_schema_test(edit_cell_tool["parameters"])

    prompt = """
    Can you help me edit a Jupyter notebook? I want to add a new markdown cell at the beginning
    of my notebook that explains the purpose of the notebook.
    """

    request_data = create_tool_request([edit_cell_tool], prompt)

    # Send the request to our API
    response = client.post("/v1/messages", json=request_data)

    logger.info(f"Response status: {response.status_code}")
    logger.info(f"Response body: {json.dumps(response.json(), indent=2)}")

    # Check if the response is successful
    assert response.status_code == 200

    # Check if enum values were properly interpreted
    response_data = response.json()
    content_text = response_data.get("content", [{}])[0].get("text", "")
    assert "cell_type" in content_text
    assert "markdown" in content_text


def test_verify_clean_gemini_schema():
    """Verify that the updated clean_gemini_schema function preserves additionalProperties"""

    # Test with a simple schema containing additionalProperties
    simple_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "object", "description": "The input to pass to the tool", "additionalProperties": True}
        },
    }

    # Get a deep copy to avoid any reference issues
    import copy

    schema_copy = copy.deepcopy(simple_schema)

    # Call the current function and log detailed output
    print("\n\nTesting clean_gemini_schema with additionalProperties:")
    print(f"Before clean: additionalProperties present: {'additionalProperties' in schema_copy['properties']['input']}")
    result = clean_gemini_schema(schema_copy)
    print(f"After clean: additionalProperties present: {'additionalProperties' in result['properties']['input']}")
    print(f"Original schema['properties']['input']: {json.dumps(simple_schema['properties']['input'], indent=2)}")
    print(f"Cleaned schema['properties']['input']: {json.dumps(result['properties']['input'], indent=2)}")

    # Check that additionalProperties is preserved
    assert "additionalProperties" in result["properties"]["input"], "additionalProperties should be preserved!"
    assert (
        result["properties"]["input"]["additionalProperties"] is True
    ), "additionalProperties value should remain True!"


@pytest.mark.integration
def test_fix_clean_gemini_schema():
    """Test an improved version of clean_gemini_schema that preserves additionalProperties"""

    # The key issue was that the old implementation removed additionalProperties
    # Let's verify that the current implementation preserves it
    def inspect_current_impl():
        test_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "description": "The input to pass to the tool",
                    "additionalProperties": True,
                }
            },
        }

        # Get a deep copy to avoid any reference issues
        import copy

        test_schema_copy = copy.deepcopy(test_schema)

        # Call the current function and log detailed output
        logger.info(
            f"Before clean: additionalProperties present: {'additionalProperties' in test_schema_copy['properties']['input']}"
        )
        result = clean_gemini_schema(test_schema_copy)
        logger.info(
            f"After clean: additionalProperties present: {'additionalProperties' in result['properties']['input']}"
        )

        # Return analysis
        return {
            "preserves_additionalProperties": "additionalProperties" in result["properties"]["input"],
            "detailed_result": result,
        }

    # An improved version that would keep additionalProperties
    def improved_clean_gemini_schema(schema: Any) -> Any:
        """Improved schema cleaner that preserves additionalProperties"""
        if isinstance(schema, dict):
            # Create a deep copy to avoid any reference issues
            import copy

            cleaned_schema = copy.deepcopy(schema)

            # Remove $schema if present
            if "$schema" in cleaned_schema:
                cleaned_schema.pop("$schema")

            # DO NOT remove additionalProperties

            # Handle string formats
            if cleaned_schema.get("type") == "string" and "format" in cleaned_schema:
                if cleaned_schema["format"] not in {"enum", "date-time"}:
                    cleaned_schema.pop("format")

            # Recursively clean nested structures
            for key, value in list(cleaned_schema.items()):
                if isinstance(value, (dict, list)):
                    cleaned_schema[key] = improved_clean_gemini_schema(value)

            return cleaned_schema
        elif isinstance(schema, list):
            return [improved_clean_gemini_schema(item) for item in schema]
        else:
            return schema

    # First, inspect the behavior of the current implementation
    current_behavior = inspect_current_impl()
    logger.info(f"Current implementation behavior: {json.dumps(current_behavior, indent=2)}")

    # Create a simple test case
    simple_tool = {
        "name": "TestTool",
        "description": "Simple test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "description": "Input with additionalProperties",
                    "additionalProperties": True,
                }
            },
        },
    }

    # Apply both functions and compare results
    original_result = clean_gemini_schema(simple_tool["parameters"])
    improved_result = improved_clean_gemini_schema(simple_tool["parameters"])

    logger.info(f"Original implementation result:\n{json.dumps(original_result, indent=2)}")
    logger.info(f"Improved implementation result:\n{json.dumps(improved_result, indent=2)}")

    # Check if our improved version preserves additionalProperties
    has_props = "additionalProperties" in improved_result["properties"]["input"]
    print(f"\n\nTEST RESULT - Improved implementation preserves additionalProperties: {has_props}")
    print(f"Original: {json.dumps(original_result['properties']['input'], indent=2)}")
    print(f"Improved: {json.dumps(improved_result['properties']['input'], indent=2)}")

    if has_props:
        print("SUCCESS: Improved implementation preserves additionalProperties")
    else:
        print("FAILED: Improved implementation still removes additionalProperties")

    # Now let's test the BatchTool specifically
    batch_tool = {
        "name": "BatchTool",
        "description": "Batch execution tool that runs multiple tools",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "A short description of the batch operation"},
                "invocations": {
                    "type": "array",
                    "description": "The list of tool invocations to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "The name of the tool to invoke"},
                            "input": {
                                "type": "object",
                                "description": "The input to pass to the tool",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["tool_name", "input"],
                    },
                },
            },
            "required": ["description", "invocations"],
        },
    }

    # Proposed implementation for the actual codebase
    def proposed_clean_gemini_schema_implementation(schema: Any) -> Any:
        """
        Minimal schema cleaner for Gemini compatibility.
        Preserves additionalProperties and only removes what's absolutely necessary.
        """
        if isinstance(schema, dict):
            cleaned_schema = schema.copy()  # Use copy() for shallow copy

            # Remove $schema if present - this is the only field we know needs removal
            if "$schema" in cleaned_schema:
                cleaned_schema.pop("$schema")

            # DO NOT remove or transform additionalProperties

            # Only handle known problematic formats for string types
            if cleaned_schema.get("type") == "string" and "format" in cleaned_schema:
                if cleaned_schema["format"] not in {"enum", "date-time"}:
                    cleaned_schema.pop("format")

            # Handle null values in required lists
            if "required" in cleaned_schema and (
                cleaned_schema["required"] is None
                or (isinstance(cleaned_schema["required"], list) and not cleaned_schema["required"])
            ):
                cleaned_schema.pop("required")

            # Recursively process nested structures
            for key, value in list(cleaned_schema.items()):
                if isinstance(value, (dict, list)):
                    cleaned_schema[key] = proposed_clean_gemini_schema_implementation(value)

            return cleaned_schema
        elif isinstance(schema, list):
            # Remove null items from lists
            return [proposed_clean_gemini_schema_implementation(item) for item in schema if item is not None]
        else:
            return schema

    # Test our final proposed implementation
    proposed_result = proposed_clean_gemini_schema_implementation(batch_tool["parameters"])
    logger.info(f"Proposed implementation result:\n{json.dumps(proposed_result, indent=2)}")

    # Check nested additionalProperties
    has_additional_props = (
        "additionalProperties" in proposed_result["properties"]["invocations"]["items"]["properties"]["input"]
    )
    print("\nProposed implementation result for BatchTool:")
    print(json.dumps(proposed_result["properties"]["invocations"]["items"]["properties"]["input"], indent=2))
    print(f"Preserves nested additionalProperties: {has_additional_props}")

    # This is our recommended implementation for src/converters.py
    logger.info("\nRECOMMENDED IMPLEMENTATION:")
    logger.info(
        """
    def clean_gemini_schema(schema: Any) -> Any:
        \"\"\"
        Minimal schema cleaner for Gemini compatibility.
        Preserves additionalProperties and only removes what's absolutely necessary.
        \"\"\"
        if isinstance(schema, dict):
            cleaned_schema = schema.copy()  # Use copy() for shallow copy

            # Remove $schema if present - this is the only field we know needs removal
            if "$schema" in cleaned_schema:
                cleaned_schema.pop("$schema")

            # DO NOT remove or transform additionalProperties

            # Only handle known problematic formats for string types
            if cleaned_schema.get("type") == "string" and "format" in cleaned_schema:
                if cleaned_schema["format"] not in {"enum", "date-time"}:
                    cleaned_schema.pop("format")

            # Handle null values in required lists
            if "required" in cleaned_schema and (cleaned_schema["required"] is None or
                                              (isinstance(cleaned_schema["required"], list) and not cleaned_schema["required"])):
                cleaned_schema.pop("required")

            # Recursively process nested structures
            for key, value in list(cleaned_schema.items()):
                if isinstance(value, (dict, list)):
                    cleaned_schema[key] = clean_gemini_schema(value)

            return cleaned_schema
        elif isinstance(schema, list):
            # Remove null items from lists
            return [clean_gemini_schema(item) for item in schema if item is not None]
        else:
            return schema
    """
    )
