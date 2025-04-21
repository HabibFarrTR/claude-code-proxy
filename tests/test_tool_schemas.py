import json
import logging
from typing import Dict, List

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
    """Verify that the clean_gemini_schema function transforms additionalProperties correctly"""

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

    # Check that additionalProperties is transformed into _additionalProps
    assert "additionalProperties" not in result["properties"]["input"], "additionalProperties should be removed"
    assert "_additionalProps" in result["properties"]["input"]["properties"], "_additionalProps should be added"
    assert (
        result["properties"]["input"]["properties"]["_additionalProps"]["type"] == "object"
    ), "_additionalProps should have type 'object'"
    assert (
        "properties" in result["properties"]["input"]["properties"]["_additionalProps"]
    ), "_additionalProps should have properties"
    assert (
        "*" in result["properties"]["input"]["properties"]["_additionalProps"]["properties"]
    ), "_additionalProps should have wildcard property"


@pytest.mark.integration
def test_current_implementation_assessment():
    """Evaluate the current implementation of clean_gemini_schema for schema handling"""

    # Test if the current implementation is working correctly with various schema patterns
    def evaluate_current_implementation():
        # Test 1: Simple additionalProperties: true
        test_schema_1 = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "description": "The input to pass to the tool",
                    "additionalProperties": True,
                }
            },
        }

        # Test 2: Nested additionalProperties
        test_schema_2 = {
            "type": "object",
            "properties": {
                "invocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "object",
                                "additionalProperties": True,
                            }
                        },
                    },
                }
            },
        }

        # Process both test schemas
        import copy

        result_1 = clean_gemini_schema(copy.deepcopy(test_schema_1))
        result_2 = clean_gemini_schema(copy.deepcopy(test_schema_2))

        logger.info("Test 1 result: Simple additionalProperties: true")
        logger.info(
            f"Original input.additionalProperties exists: {'additionalProperties' in test_schema_1['properties']['input']}"
        )
        logger.info(f"Transformed: {'_additionalProps' in result_1['properties']['input'].get('properties', {})}")

        logger.info("\nTest 2 result: Nested additionalProperties")
        nested_input = result_2["properties"]["invocations"]["items"]["properties"]["input"]
        logger.info(
            f"Transformed nested _additionalProps exists: {'_additionalProps' in nested_input.get('properties', {})}"
        )

        return {
            "test1_transforms_additionalProps": "_additionalProps"
            in result_1["properties"]["input"].get("properties", {}),
            "test2_transforms_nested": "_additionalProps" in nested_input.get("properties", {}),
        }

    # Run the evaluation
    evaluation = evaluate_current_implementation()
    logger.info(f"Evaluation results: {json.dumps(evaluation, indent=2)}")

    # Create a test case with the BatchTool schema
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

    # Process the BatchTool schema
    batch_result = clean_gemini_schema(batch_tool["parameters"])

    # Examine the results for BatchTool
    batch_input_path = batch_result["properties"]["invocations"]["items"]["properties"]["input"]
    batch_has_transform = "_additionalProps" in batch_input_path.get("properties", {})

    print("\nCurrent Implementation BatchTool Result:")
    print(json.dumps(batch_input_path, indent=2))
    print(f"Transforms nested additionalProperties: {batch_has_transform}")

    # Verify the transformation is correct
    assert "additionalProperties" not in batch_input_path, "additionalProperties should be removed"
    assert "_additionalProps" in batch_input_path.get("properties", {}), "_additionalProps should be added"
    assert "*" in batch_input_path["properties"]["_additionalProps"]["properties"], "Wildcard property should be added"

    # Log success if test passes
    logger.info("Current implementation is working correctly for BatchTool schema.")
    print("Current implementation successfully transforms BatchTool schema as expected.")
