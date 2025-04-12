I am trying to implement a proxy so i can use you as model behind  with claude code...
I am doing this because we have internal access to Gemini and want to have the option to use it as the model behind claude code.
However we need to auth a particular way:
```python
import requests
import json
from google.oauth2.credentials import Credentials as OAuth2Credentials
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Image

# Set your workspace id
workspace_id = "PracticalLawxOhJ"  # Open workspace console to get your workspace_id

# Set your model name, for available model names please refer documentation
model_name = "gemini-2.5-pro-preview-03-25"

# Create a dictionary payload with workspace_id and model_name
payload = {
    "workspace_id": workspace_id,
    "model_name": model_name,
}

# Define the URL for the request
url = "https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"

# Send a POST request to the URL with headers and the payload
resp = requests.post(url, headers=None, json=payload)

# Load the response content as JSON
credentials = json.loads(resp.content)
# Check if the credentials contain a token
if "token" in credentials:
    success_flag = 1
else:
    success_flag = 0
    print("Incorrect Workspace ID or Model Name")

if success_flag == 1:
    project_id = credentials["project_id"]
    location = credentials["region"]
    temp_creds = OAuth2Credentials(credentials["token"])
    print(f"-----Gemini Credentials are fetched and are stored in variables, these are valid till {credentials['expires_on']}-----")

    # Initialize Vertex AI with the impersonated credentials
    vertexai.init(project=project_id, location=location, credentials=temp_creds)

    model = GenerativeModel(model_name)
    chat = model.start_chat(response_validation=False)
    prompt = "Hello?"
    response = chat.send_message(prompt)
    print(response.text)
```
The bug is that from claude code ui sometime I get the right answer and sometimes something like this: 
```python
$ ANTHROPIC_BASE_URL=http://localhost:8082 claude
╭────────────────────────────────────────────╮
│ ✻ Welcome to Claude Code research preview! │
│                                            │
│   /help for help                           │
│                                            │
│   cwd: /Users/a6133790/workspace/llm-chat  │
│                                            │
│   ──────────────────────────────────────── │
│                                            │
│   Overrides (via env):                     │
│                                            │
│   • API Base URL: http://localhost:8082    │
╰────────────────────────────────────────────╯

 ✔ Loaded project + user memory • /memory

> what is going on in this project

⏺ "print(ReadFileTool(paths=["README.md"]).model_dump_json())

> what is going on in this project

⏺ "print(FileReaderTool().read_file(file_path="README.md"))

> what is going on in this project

⏺ "print(ReadFileTool().run(paths=["README.md"]))

> what is going on in this project

⏺ This project is a Python command-line chat application, likely named "chat-gemini", built using Poetry for dependency management. It includes components for
  handling chat logic, authentication, and file reading, with tests and linting/formatting tools (pytest, ruff, black, mypy).
```
Which my guess is has to do with how the streaming is handled.
The code was originally a fork of https://github.com/1rgs/claude-code-proxy which uses LiteLLM https://github.com/BerriAI/litellm
but since i couldn't make LiteLLM work with our internal auth, I decided to remove LiteLLM and write my own streaming code.
It has been fun, but I am not sure if I am doing it right.
You can find the original LiteLLM code at /Users/a6133790/workspace/claude-code-proxy-original/server.py
I wonder if you spot the issue specially by comparing my solution with the original LiteLLM code.