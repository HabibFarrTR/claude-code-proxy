#!/usr/bin/env bash
# Claude Proxy shell integration
# Add to your .bashrc or .zshrc with: source /path/to/claude-profile.sh

# Configuration (modify as needed)
CLAUDE_PROXY_DIR="$HOME/.local/claude-proxy"  # Default installation directory
CLAUDE_PROXY_PORT=8082                        # Default port

# Function to start Claude with native API
claude_native() {
  echo "🧠 Starting Claude Code with native Anthropic API..."
  unset ANTHROPIC_BASE_URL
  claude "$@"
}

# Function to start Claude with Gemini proxy
claude_gemini() {
  local port=${1:-$CLAUDE_PROXY_PORT}
  shift 2>/dev/null || true

  if ! is_proxy_running "$port"; then
    echo "🚀 Starting Gemini proxy on port $port..."
    start_proxy "$port"
    sleep 2 # Give proxy time to start
  fi

  echo "🤖 Starting Claude Code with Gemini backend on port $port..."
  ANTHROPIC_BASE_URL="http://localhost:$port" claude "$@"
}

# Function to check if proxy is running
is_proxy_running() {
  local port=${1:-$CLAUDE_PROXY_PORT}
  lsof -i ":$port" &>/dev/null
  return $?
}

# Function to start the proxy
start_proxy() {
  local port=${1:-$CLAUDE_PROXY_PORT}

  # Use the installed proxy if available
  if command -v claude-proxy &>/dev/null; then
    claude-proxy start "$port" > /tmp/claude-proxy.log 2>&1 &
  # Otherwise try to find it in the source directory
  elif [ -d "$CLAUDE_PROXY_DIR" ]; then
    (cd "$CLAUDE_PROXY_DIR" && poetry run uvicorn src.server:app --host 0.0.0.0 --port "$port" > /tmp/claude-proxy.log 2>&1 &)
  else
    echo "❌ Error: Proxy not found. Install with scripts/install.sh or set CLAUDE_PROXY_DIR"
    return 1
  fi

  echo "✅ Proxy started on port $port (PID: $!)"
}

# Function to stop the proxy
stop_proxy() {
  local port=${1:-$CLAUDE_PROXY_PORT}
  local pid=$(lsof -ti ":$port")
  if [ -n "$pid" ]; then
    echo "🛑 Stopping proxy on port $port (PID: $pid)..."
    kill "$pid"
  else
    echo "⚠️ No proxy running on port $port"
  fi
}

# Aliases
alias claude-orig="claude_native"
alias claude-gem="claude_gemini"
alias proxy-start="start_proxy"
alias proxy-stop="stop_proxy"
alias proxy-status="is_proxy_running && echo '✅ Proxy is running' || echo '❌ Proxy is not running'"

# Print welcome message if this file is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Claude Proxy shell integration"
  echo ""
  echo "This script should be sourced in your shell profile, not executed directly:"
  echo ""
  echo "  source ${BASH_SOURCE[0]}"
  echo ""
  echo "Once sourced, you can use:"
  echo "  claude-orig     # Use Claude with native API"
  echo "  claude-gem      # Use Claude with Gemini proxy"
  echo "  proxy-start     # Start the proxy"
  echo "  proxy-stop      # Stop the proxy"
  echo "  proxy-status    # Check if proxy is running"
  echo ""
fi
