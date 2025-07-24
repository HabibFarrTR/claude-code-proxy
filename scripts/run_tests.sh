#!/usr/bin/env bash

# Test runner script for the Claude Code Proxy
# This script starts the server before running tests and cleans up afterward

set -e  # Exit on any error

# Default port for the server
PORT=8083

# Parse command line arguments
TEST_FILTER=""
SKIP_AUTH=false
VERBOSE=false

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help             Show this help message"
    echo "  -t, --test TEST        Run a specific test (e.g. test_anthropic, test_aiplatform)"
    echo "  --skip-auth            Skip auth tests (useful if you don't have credentials)"
    echo "  -v, --verbose          Show full output of server logs"
    echo
    echo "Examples:"
    echo "  $0                      # Run all tests"
    echo "  $0 -t test_anthropic    # Run only Anthropic tests"
    echo "  $0 -t test_aiplatform   # Run only AI Platform tests"
    echo "  $0 --skip-auth          # Skip all authentication tests"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -t|--test)
            TEST_FILTER="::$2"
            shift 2
            ;;
        --skip-auth)
            SKIP_AUTH=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
    fi

    # Kill any leftover uvicorn processes on our port
    LEFTOVER_PID=$(lsof -i:$PORT -t 2>/dev/null || true)
    if [ -n "$LEFTOVER_PID" ]; then
        echo "Killing leftover process on port $PORT (PID: $LEFTOVER_PID)..."
        kill $LEFTOVER_PID 2>/dev/null || true
    fi

    echo "Cleanup complete"
}

# Register cleanup function to run on script exit
trap cleanup EXIT

# Create a temporary log file for the server
SERVER_LOG=$(mktemp)
echo "Server logs will be saved to: $SERVER_LOG"

# Ensure port is available
EXISTING_PID=$(lsof -i:$PORT -t 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Port $PORT is already in use by process $EXISTING_PID. Stopping it..."
    kill $EXISTING_PID 2>/dev/null || true
    sleep 1
fi

# Add a note about running mltools-cli aws-login if testing AI Platform
if [[ "$TEST_FILTER" == "::test_aiplatform" ]] && [[ "$SKIP_AUTH" == "false" ]]; then
    echo ""
    echo "🔑 IMPORTANT: To test AI Platform, you need to run 'mltools-cli aws-login' first"
    echo "   This is required to set up AWS credentials for Thomson Reuters authentication"
    echo ""

    # Automatically assume yes for now since the user has already logged in
    REPLY="y"
fi

echo "Starting server on port $PORT..."
poetry run uvicorn src.server:app --host 0.0.0.0 --port $PORT > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start (PID: $SERVER_PID)..."
for i in {1..10}; do
    if curl -s http://localhost:$PORT/ >/dev/null 2>&1; then
        echo "Server is up and running"
        break
    fi

    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Server process died!"
        if [ "$VERBOSE" == "true" ]; then
            cat $SERVER_LOG
        else
            echo "Last 20 lines of server log:"
            tail -20 $SERVER_LOG
        fi
        exit 1
    fi

    echo "Waiting... ($i/10)"
    sleep 1
done

# Check if server is actually running
if ! curl -s http://localhost:$PORT/ >/dev/null 2>&1; then
    echo "Server failed to start properly!"
    if [ "$VERBOSE" == "true" ]; then
        cat $SERVER_LOG
    else
        echo "Last 20 lines of server log:"
        tail -20 $SERVER_LOG
    fi
    exit 1
fi

# Modify the test file to use the correct port
echo "Updating test file to use port $PORT..."
sed -i.bak "s|PROXY_API_URL = \"http://localhost:[0-9]\+/v1/messages\"|PROXY_API_URL = \"http://localhost:$PORT/v1/messages\"|g" tests/test_server.py
rm -f tests/test_server.py.bak

# Run the tests
echo "Running tests..."
if [ "$SKIP_AUTH" == "true" ]; then
    if [ -n "$TEST_FILTER" ]; then
        # If a specific test was requested but we want to skip auth tests
        if [[ "$TEST_FILTER" == "::test_aiplatform" ]]; then
            echo "You requested AI Platform tests but also --skip-auth. These options are incompatible."
            exit 1
        fi
        # Run the specific test
        TEST_CMD="poetry run pytest -xvs tests/test_server.py${TEST_FILTER}"
    else
        # Skip all auth tests
        echo "Skipping authentication tests..."
        TEST_CMD="poetry run pytest -xvs tests/test_server.py -k 'not test_aiplatform'"
    fi
else
    # Run all requested tests
    TEST_CMD="poetry run pytest -xvs tests/test_server.py${TEST_FILTER}"
fi

echo "Executing: $TEST_CMD"

# Actually run the tests
if $TEST_CMD; then
    echo "✅ All tests passed!"
    TEST_SUCCESS=true
else
    echo "❌ Some tests failed!"
    TEST_SUCCESS=false
fi

# Output server logs on failure or if in verbose mode
if [ "$TEST_SUCCESS" == "false" ] || [ "$VERBOSE" == "true" ]; then
    echo "Server log:"
    cat $SERVER_LOG
fi

# Clean up temporary log file
rm -f $SERVER_LOG

if [ "$TEST_SUCCESS" == "true" ]; then
    exit 0
else
    exit 1
fi
