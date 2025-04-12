#!/usr/bin/env bash
# Claude Proxy Installer
# Installs claude-code-proxy without requiring the full source repository

set -e  # Exit on error

# Default installation directory
DEFAULT_INSTALL_DIR="$HOME/.local/claude-proxy"
DEFAULT_BIN_DIR="$HOME/.local/bin"

# Repository configuration - update these when you host the repo publicly
# For now, we use local file creation instead of downloading
USE_GITHUB=false
GITHUB_REPO="https://github.com/YOUR-COMPANY/claude-code-proxy"
VERSION="main"  # Could be a specific tag/release

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "┌──────────────────────────────────────────┐"
    echo "│  Claude Code Proxy Installer              │"
    echo "│  Anthropic API to Thomson Reuters Bridge  │"
    echo "└──────────────────────────────────────────┘"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse command line arguments
parse_args() {
    INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    BIN_DIR="$DEFAULT_BIN_DIR"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --bin)
                BIN_DIR="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dir DIR    Installation directory (default: $DEFAULT_INSTALL_DIR)"
                echo "  --bin DIR    Executable directory (default: $DEFAULT_BIN_DIR)"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

check_dependencies() {
    print_info "Checking dependencies..."

    MISSING_DEPS=()

    for cmd in python3 pip curl unzip; do
        if ! command -v $cmd &> /dev/null; then
            MISSING_DEPS+=($cmd)
        fi
    done

    if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
        print_error "Missing required dependencies: ${MISSING_DEPS[*]}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi

    # Check Python version
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

    if [[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 8) ]]; then
        print_error "Python 3.8 or higher is required. Found: $PY_VERSION"
        exit 1
    fi

    print_success "All dependencies satisfied!"
}

create_virtualenv() {
    print_info "Creating virtual environment in $INSTALL_DIR..."

    # Create installation directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"

    # Create virtual environment
    python3 -m venv "$INSTALL_DIR/venv"

    print_success "Virtual environment created!"
}

install_dependencies() {
    print_info "Installing dependencies..."

    # Activate virtual environment
    source "$INSTALL_DIR/venv/bin/activate"

    # Install required packages
    pip install --upgrade pip
    pip install fastapi uvicorn pydantic python-dotenv httpx vertexai google-auth

    print_success "Dependencies installed!"
}

create_directory_structure() {
    print_info "Creating directory structure..."

    # Create necessary directories
    mkdir -p "$INSTALL_DIR/src"

    print_success "Directory structure created!"
}

download_core_files() {
    print_info "Downloading core files..."

    # Create a temporary directory
    TMP_DIR=$(mktemp -d)

    # Function to download a single file from GitHub
    download_file() {
        local file_path="$1"
        local target_dir="$2"
        local url="$GITHUB_REPO/raw/$VERSION/$file_path"

        mkdir -p "$(dirname "$target_dir/$file_path")"

        # Add verbose output for debugging
        print_info "Downloading $file_path from $url"

        # Add User-Agent header to avoid getting HTML error pages
        curl -s -L -o "$target_dir/$file_path" -H "User-Agent: Mozilla/5.0" "$url"

        # Check if the downloaded file looks like HTML (likely an error page)
        if grep -q "<!DOCTYPE html>" "$target_dir/$file_path" 2>/dev/null; then
            print_warning "Warning: $file_path appears to contain HTML. Download might have failed."

            # For .env.example, we'll handle it in setup_environment()
            if [[ "$file_path" != ".env.example" ]]; then
                print_info "Creating empty file for $file_path"
                mkdir -p "$(dirname "$target_dir/$file_path")"
                touch "$target_dir/$file_path"
            fi
        fi
    }

    if [[ "$USE_GITHUB" == "true" ]]; then
        # List of files to download
        FILES=(
            "src/__init__.py"
            "src/server.py"
            "src/models.py"
            "src/api.py"
            "src/utils.py"
            "src/streaming.py"
            "src/authenticator.py"
            ".env.example"
            "README.md"
        )

        # Download each file
        for file in "${FILES[@]}"; do
            download_file "$file" "$INSTALL_DIR"
        done
    else
        # Create files locally instead of downloading
        print_info "GitHub download disabled. Creating files locally..."

        # Create Python package directory
        mkdir -p "$INSTALL_DIR/src"
        touch "$INSTALL_DIR/src/__init__.py"

        # Copy from local source if available
        if [[ -d "$(dirname "$0")/../src" ]]; then
            print_info "Found local source files, copying..."
            cp -R "$(dirname "$0")/../src/"* "$INSTALL_DIR/src/"
            cp "$(dirname "$0")/../.env.example" "$INSTALL_DIR/" 2>/dev/null || true
            cp "$(dirname "$0")/../README.md" "$INSTALL_DIR/" 2>/dev/null || true
        else
            # Create minimal placeholder files
            for file in "server.py" "models.py" "api.py" "utils.py" "streaming.py" "authenticator.py"; do
                if [[ ! -f "$INSTALL_DIR/src/$file" ]]; then
                    echo "# Placeholder for $file - requires manual download" > "$INSTALL_DIR/src/$file"
                fi
            done

            # Copy .env.example if we have it locally
            if [[ -f "$(dirname "$0")/../.env.example" ]]; then
                cp "$(dirname "$0")/../.env.example" "$INSTALL_DIR/"
            fi
        fi
    fi

    # Clean up temporary directory
    rm -rf "$TMP_DIR"

    print_success "Core files setup complete!"
}

create_executable() {
    print_info "Creating executable script in $BIN_DIR..."

    # Create bin directory if it doesn't exist
    mkdir -p "$BIN_DIR"

    # Create the executable
    cat > "$BIN_DIR/claude-proxy" << EOF
#!/usr/bin/env bash
# Claude Proxy executable

INSTALL_DIR="$INSTALL_DIR"
VENV="\$INSTALL_DIR/venv"
PORT="8082"
LOG_FILE="/tmp/claude-proxy.log"

# Check Python requirements
check_python_requirements() {
    # Activate virtual environment
    source "\$VENV/bin/activate" || {
        echo "❌ Error: Failed to activate virtual environment."
        echo "   Try reinstalling with 'curl -s https://raw.githubusercontent.com/$GITHUB_REPO/main/scripts/install.sh | bash'"
        return 1
    }

    # Check basic python modules
    python -c "import fastapi, uvicorn" 2>/dev/null || {
        echo "❌ Error: Required Python packages not found."
        echo "   Attempting to reinstall dependencies..."
        pip install fastapi uvicorn python-dotenv google-auth
        return $?
    }

    return 0
}

# Source virtual environment
source "\$VENV/bin/activate" 2>/dev/null || {
    echo "❌ Error: Failed to activate virtual environment."
    echo "   The proxy may not have been installed correctly."
    exit 1
}

case "\$1" in
    start)
        # Get port from second argument if provided
        if [[ -n "\$2" ]]; then
            PORT="\$2"
        fi

        # Check if already running
        if lsof -ti ":\$PORT" &>/dev/null; then
            echo "⚠️ Proxy is already running on port \$PORT"
            echo "   Run 'claude-proxy status \$PORT' to see details"
            exit 0
        fi

        # Check Python requirements
        check_python_requirements || exit 1

        # Check for .env file and offer to create if missing
        if [[ ! -f "\$INSTALL_DIR/.env" ]]; then
            echo "⚠️ No .env file found. Creating from template..."
            if [[ -f "\$INSTALL_DIR/.env.example" ]] && ! grep -q "<!DOCTYPE html>" "\$INSTALL_DIR/.env.example"; then
                cp "\$INSTALL_DIR/.env.example" "\$INSTALL_DIR/.env"
            else
                # Create a minimal .env file
                cat > "\$INSTALL_DIR/.env" << ENVEOF
# Thomson Reuters AIplatform Configuration
# IMPORTANT: You must run 'mltools-cli aws-login' to set up AWS credentials before using this proxy

# Required settings
WORKSPACE_ID="your-workspace-id" # Replace with your actual workspace ID

# Optional settings with defaults
AUTH_URL="https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"

# The proxy will automatically map:
# - claude-3-sonnet and claude-3-opus → gemini-2.5-pro-preview-03-25
# - claude-3-haiku → gemini-2.0-flash
ENVEOF
            fi
            echo "⚠️ Please edit \$INSTALL_DIR/.env with your configuration"
        fi

        echo "🚀 Starting claude-code-proxy on port \$PORT..."
        echo "   Logs will be written to \$LOG_FILE"
        cd "\$INSTALL_DIR" && \
        {
            # Start with a timestamp in log
            echo "=== Starting Claude Proxy on port \$PORT at \$(date) ===" > "\$LOG_FILE"
            python -m uvicorn src.server:app --host 0.0.0.0 --port "\$PORT" \${@:3} >> "\$LOG_FILE" 2>&1 &
            PID=\$!

            # Check if process is still running after 2 seconds
            sleep 2
            if kill -0 \$PID 2>/dev/null; then
                echo "✅ Proxy started successfully (PID: \$PID)"
                echo "   Use 'claudex --gemini' to run Claude with this proxy"
                echo "   Use 'claude-proxy status' to check status"
                echo "   Use 'claude-proxy stop' to stop the proxy"
            else
                echo "❌ Proxy failed to start. Check \$LOG_FILE for errors."
                exit 1
            fi
        }
        ;;
    stop)
        PORT="\${2:-\$PORT}"
        PID=\$(lsof -ti ":\$PORT" 2>/dev/null)
        if [[ -n "\$PID" ]]; then
            echo "🛑 Stopping proxy running on port \$PORT (PID: \$PID)..."
            kill "\$PID"

            # Verify process was stopped
            sleep 1
            if kill -0 \$PID 2>/dev/null; then
                echo "⚠️ Process is still running. Attempting force kill..."
                kill -9 "\$PID" 2>/dev/null
                sleep 1
                if kill -0 \$PID 2>/dev/null; then
                    echo "❌ Failed to stop process \$PID"
                else
                    echo "✅ Process \$PID forcefully terminated"
                fi
            else
                echo "✅ Proxy stopped successfully"
            fi
        else
            echo "⚠️ No proxy running on port \$PORT"
        fi
        ;;
    status)
        PORT="\${2:-\$PORT}"
        PID=\$(lsof -ti ":\$PORT" 2>/dev/null)
        if [[ -n "\$PID" ]]; then
            RUNTIME=\$(ps -o etime= -p "\$PID" 2>/dev/null)
            echo "✅ Proxy is running on port \$PORT"
            echo "   PID: \$PID"
            echo "   Uptime: \${RUNTIME:-unknown}"
            echo "   Log file: \$LOG_FILE"

            # Show latest log entries
            if [[ -f "\$LOG_FILE" ]]; then
                echo ""
                echo "Last 5 log entries:"
                tail -5 "\$LOG_FILE"
            fi
        else
            echo "❌ Proxy is not running on port \$PORT"
        fi
        ;;
    logs)
        if [[ -f "\$LOG_FILE" ]]; then
            if command -v tail &>/dev/null; then
                echo "📜 Showing logs from \$LOG_FILE (Ctrl+C to exit):"
                echo ""
                tail -f "\$LOG_FILE"
            else
                echo "📜 Log file location: \$LOG_FILE"
            fi
        else
            echo "❌ Log file not found at \$LOG_FILE"
        fi
        ;;
    setup-env)
        if [[ ! -f "\$INSTALL_DIR/.env" ]]; then
            # Check if .env.example exists and is valid (not HTML)
            if [[ -f "\$INSTALL_DIR/.env.example" ]] && ! grep -q "<!DOCTYPE html>" "\$INSTALL_DIR/.env.example"; then
                cp "\$INSTALL_DIR/.env.example" "\$INSTALL_DIR/.env"
                echo "📝 Created .env file from template."
            else
                # Create a default .env file
                cat > "\$INSTALL_DIR/.env" << ENVEOF
# Thomson Reuters AIplatform Configuration
# IMPORTANT: You must run 'mltools-cli aws-login' to set up AWS credentials before using this proxy

# Required settings
WORKSPACE_ID="your-workspace-id" # Replace with your actual workspace ID

# Optional settings with defaults
AUTH_URL="https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"

# The proxy will automatically map:
# - claude-3-sonnet and claude-3-opus → gemini-2.5-pro-preview-03-25
# - claude-3-haiku → gemini-2.0-flash
ENVEOF
                echo "📝 Created default .env file."
            fi
            echo "⚠️ Please edit \$INSTALL_DIR/.env and set your configuration."
        else
            # Check if existing .env file is valid or is HTML
            if grep -q "<!DOCTYPE html>" "\$INSTALL_DIR/.env"; then
                echo "⚠️ Existing .env file contains HTML. Creating new one..."
                mv "\$INSTALL_DIR/.env" "\$INSTALL_DIR/.env.broken"
                # Create a default .env file
                cat > "\$INSTALL_DIR/.env" << ENVEOF
# Thomson Reuters AIplatform Configuration
# IMPORTANT: You must run 'mltools-cli aws-login' to set up AWS credentials before using this proxy

# Required settings
WORKSPACE_ID="your-workspace-id" # Replace with your actual workspace ID

# Optional settings with defaults
AUTH_URL="https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"

# The proxy will automatically map:
# - claude-3-sonnet and claude-3-opus → gemini-2.5-pro-preview-03-25
# - claude-3-haiku → gemini-2.0-flash
ENVEOF
                echo "📝 Created new .env file. Old file saved as .env.broken"
                echo "⚠️ Please edit \$INSTALL_DIR/.env and set your configuration."
            else
                echo "ℹ️ .env file already exists."
                echo "   To edit: nano \$INSTALL_DIR/.env"
            fi
        fi
        ;;
    update)
        echo "🔄 Updating claude-proxy..."
        # Re-run the installer script
        curl -s -L -H "User-Agent: Mozilla/5.0" https://raw.githubusercontent.com/your-company/claude-code-proxy-installer/main/scripts/install.sh | bash
        ;;
    *)
        echo "Claude Proxy - Thomson Reuters AI Platform Bridge"
        echo ""
        echo "Usage: claude-proxy COMMAND [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  start [PORT]    Start the proxy server (default port: 8082)"
        echo "  stop [PORT]     Stop the proxy server"
        echo "  status [PORT]   Check if the proxy is running"
        echo "  logs            View the proxy server logs"
        echo "  setup-env       Create or fix .env file from template"
        echo "  update          Update to the latest version"
        echo ""
        echo "Examples:"
        echo "  claude-proxy start             # Start on default port 8082"
        echo "  claude-proxy start 8083        # Start on custom port 8083"
        echo "  claude-proxy stop              # Stop the default instance"
        echo "  claude-proxy logs              # View the logs"
        echo ""
        echo "Installed in: \$INSTALL_DIR"
        echo "To use with Claude Code: claudex --gemini"
        ;;
esac
EOF

    # Make it executable
    chmod +x "$BIN_DIR/claude-proxy"

    print_success "Executable script created: $BIN_DIR/claude-proxy"
}

create_claude_wrapper() {
    print_info "Creating Claude wrapper for easy switching..."

    # Create the Claude wrapper
    cat > "$BIN_DIR/claudex" << EOF
#!/usr/bin/env bash
# Claude Code wrapper with provider switching

DEFAULT_PORT="8082"

# Function to check if proxy is running
is_proxy_running() {
    local port="\${1:-\$DEFAULT_PORT}"
    lsof -ti ":\$port" &>/dev/null
    return \$?
}

# Function to ensure Claude command is available
check_claude_available() {
    if ! command -v claude &> /dev/null; then
        echo "❌ Error: 'claude' command not found."
        echo "Please install Claude Code CLI first with 'pip install anthropic-claude-cli'"
        return 1
    fi
    return 0
}

# Handle arguments
if [[ "\$1" == "--gemini" || "\$1" == "-g" ]]; then
    # Check if Claude is available
    if ! check_claude_available; then
        exit 1
    fi

    # Extract port if provided
    PORT="\$DEFAULT_PORT"
    if [[ "\$2" =~ ^[0-9]+$ ]]; then
        PORT="\$2"
        shift 2
    else
        shift 1
    fi

    # Check if proxy is running, start if not
    if ! is_proxy_running "\$PORT"; then
        echo "🚀 Starting proxy on port \$PORT..."
        claude-proxy start "\$PORT" &>/dev/null &

        # Wait for proxy to start (with timeout)
        MAX_WAIT=10
        WAIT_TIME=0
        echo -n "Waiting for proxy to start "
        while ! is_proxy_running "\$PORT" && [ \$WAIT_TIME -lt \$MAX_WAIT ]; do
            echo -n "."
            sleep 1
            WAIT_TIME=\$((WAIT_TIME + 1))
        done
        echo ""

        if ! is_proxy_running "\$PORT"; then
            echo "❌ Failed to start proxy within \$MAX_WAIT seconds."
            echo "Check logs at /tmp/claude-proxy.log"
            exit 1
        fi

        echo "✅ Proxy started successfully!"
    fi

    # Run Claude with proxy
    echo "🔄 Running Claude with Gemini proxy (port: \$PORT)..."
    ANTHROPIC_BASE_URL="http://localhost:\$PORT" claude "\$@"

elif [[ "\$1" == "--status" || "\$1" == "-s" ]]; then
    # Check proxy status
    PORT="\${2:-\$DEFAULT_PORT}"
    if is_proxy_running "\$PORT"; then
        PID=\$(lsof -ti ":\$PORT")
        echo "✅ Proxy is running on port \$PORT (PID: \$PID)"
    else
        echo "❌ Proxy is not running on port \$PORT"
    fi

elif [[ "\$1" == "--stop" ]]; then
    # Stop the proxy
    PORT="\${2:-\$DEFAULT_PORT}"
    PID=\$(lsof -ti ":\$PORT")
    if [ -n "\$PID" ]; then
        echo "🛑 Stopping proxy on port \$PORT (PID: \$PID)..."
        kill "\$PID"
    else
        echo "⚠️ No proxy running on port \$PORT"
    fi

elif [[ "\$1" == "--help" || "\$1" == "-h" ]]; then
    echo "Claude Code with provider switching"
    echo ""
    echo "Usage: claudex [OPTIONS] [CLAUDE_ARGS]"
    echo ""
    echo "Options:"
    echo "  --gemini, -g [PORT]   Use Gemini proxy (default port: 8082)"
    echo "  --claude, -c          Use native Claude API (default)"
    echo "  --status, -s [PORT]   Check if proxy is running"
    echo "  --stop [PORT]         Stop the proxy"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  claudex               # Use native Claude API"
    echo "  claudex -g            # Use Gemini proxy on port 8082"
    echo "  claudex -g 8083       # Use Gemini proxy on port 8083"
    echo "  claudex -c            # Use native Claude API explicitly"
    echo "  claudex -s            # Check if proxy is running"
    echo "  claudex --stop        # Stop the proxy"
    echo ""
    echo "Note: Additional arguments are passed to Claude Code"

elif [[ "\$1" == "--claude" || "\$1" == "-c" ]]; then
    # Check if Claude is available
    if ! check_claude_available; then
        exit 1
    fi

    # Run Claude with native API
    shift 1
    echo "🧠 Running Claude with native API..."
    unset ANTHROPIC_BASE_URL
    claude "\$@"
else
    # Check if Claude is available
    if ! check_claude_available; then
        exit 1
    fi

    # Default to native Claude API
    echo "🧠 Running Claude with native API..."
    unset ANTHROPIC_BASE_URL
    claude "\$@"
fi
EOF

    # Make it executable
    chmod +x "$BIN_DIR/claudex"

    print_success "Claude wrapper created: $BIN_DIR/claudex"
}

setup_environment() {
    print_info "Setting up environment..."

    # Create initial .env file if it doesn't exist
    if [[ ! -f "$INSTALL_DIR/.env" ]]; then
        # First check if .env.example exists
        if [[ -f "$INSTALL_DIR/.env.example" ]]; then
            # Validate .env.example - check if it contains HTML or other invalid content
            if grep -q "<!DOCTYPE html>" "$INSTALL_DIR/.env.example" || \
               grep -q "<html" "$INSTALL_DIR/.env.example" || \
               ! grep -q "WORKSPACE_ID" "$INSTALL_DIR/.env.example"; then
                print_warning "Downloaded .env.example contains invalid content. Creating default .env file instead."
                CREATE_DEFAULT=true
            else
                cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
                print_info "Created .env file from template."
                CREATE_DEFAULT=false
            fi
        else
            print_warning ".env.example not found. Creating default .env file."
            CREATE_DEFAULT=true
        fi

        # Create a default .env file if needed
        if [[ "$CREATE_DEFAULT" == "true" ]]; then
            print_info "Creating default .env file..."
            cat > "$INSTALL_DIR/.env" << EOF
# Thomson Reuters AIplatform Configuration
# IMPORTANT: You must run 'mltools-cli aws-login' to set up AWS credentials before using this proxy

# Required settings
WORKSPACE_ID="your-workspace-id" # Replace with your actual workspace ID

# Optional settings with defaults
AUTH_URL="https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"
# The proxy will automatically map:
# - claude-3-sonnet and claude-3-opus → gemini-2.5-pro-preview-03-25
# - claude-3-haiku → gemini-2.0-flash
EOF
        fi

        # Validate the final .env file
        if grep -q "<!DOCTYPE html>" "$INSTALL_DIR/.env" || \
           ! grep -q "WORKSPACE_ID" "$INSTALL_DIR/.env"; then
            print_error "Failed to create a valid .env file. Please create it manually."
            cat > "$INSTALL_DIR/.env" << EOF
# Thomson Reuters AIplatform Configuration
# IMPORTANT: You must run 'mltools-cli aws-login' to set up AWS credentials before using this proxy

# Required settings
WORKSPACE_ID="your-workspace-id" # Replace with your actual workspace ID

# Optional settings with defaults
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"

# The proxy will automatically map:
# - claude-3-sonnet and claude-3-opus → gemini-2.5-pro-preview-03-25
# - claude-3-haiku → gemini-2.0-flash
EOF
        fi

        print_warning "Please edit $INSTALL_DIR/.env and set your configuration."
    else
        # Check if existing .env file is valid (not HTML)
        if grep -q "<!DOCTYPE html>" "$INSTALL_DIR/.env" || \
           ! grep -q "WORKSPACE_ID" "$INSTALL_DIR/.env"; then
            print_warning "Existing .env file contains invalid content. Creating new default .env file..."
            mv "$INSTALL_DIR/.env" "$INSTALL_DIR/.env.broken"
            print_info "Backed up invalid .env to .env.broken"

            cat > "$INSTALL_DIR/.env" << EOF
# Thomson Reuters AIplatform Configuration
# IMPORTANT: You must run 'mltools-cli aws-login' to set up AWS credentials before using this proxy

# Required settings
WORKSPACE_ID="your-workspace-id" # Replace with your actual workspace ID

# Optional settings with defaults
AUTH_URL="https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"

# The proxy will automatically map:
# - claude-3-sonnet and claude-3-opus → gemini-2.5-pro-preview-03-25
# - claude-3-haiku → gemini-2.0-flash
EOF
            print_warning "Please edit $INSTALL_DIR/.env and set your configuration."
        else
            print_info ".env file already exists and appears valid."
        fi
    fi
}

print_next_steps() {
    echo ""
    echo -e "${GREEN}Installation Complete!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Edit your configuration file:"
    echo "   $INSTALL_DIR/.env"
    echo ""
    echo "2. Make sure $BIN_DIR is in your PATH"
    echo ""
    echo "3. Start the proxy server:"
    echo "   claude-proxy start"
    echo ""
    echo "4. Use Claude with the Gemini proxy:"
    echo "   claudex --gemini"
    echo ""
    echo "5. Or use the native Claude API:"
    echo "   claudex"
    echo ""
    echo "For more information, see:"
    echo "   claudex --help"
    echo "   claude-proxy --help"
    echo ""
}

# Main installation process
main() {
    print_banner
    parse_args "$@"
    check_dependencies
    create_virtualenv
    install_dependencies
    create_directory_structure
    download_core_files
    create_executable
    create_claude_wrapper
    setup_environment
    print_next_steps
}

# Execute main function
main "$@"
