#!/usr/bin/env bash
# Claude Proxy Installer
# Installs claude-code-proxy without requiring the full source repository

set -e  # Exit on error

# Default installation directory
DEFAULT_INSTALL_DIR="$HOME/.local/claude-proxy"
DEFAULT_BIN_DIR="$HOME/.local/bin"
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
    if [[ $(echo "$PY_VERSION < 3.8" | bc) -eq 1 ]]; then
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
        curl -s -o "$target_dir/$file_path" "$url"
    }

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

    # Clean up temporary directory
    rm -rf "$TMP_DIR"

    print_success "Core files downloaded!"
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

# Source virtual environment
source "\$VENV/bin/activate"

case "\$1" in
    start)
        # Get port from second argument if provided
        if [[ -n "\$2" ]]; then
            PORT="\$2"
        fi

        echo "🚀 Starting claude-code-proxy on port \$PORT..."
        cd "\$INSTALL_DIR" && \
        python -m uvicorn src.server:app --host 0.0.0.0 --port "\$PORT" \${@:3}
        ;;
    stop)
        PORT="\${2:-\$PORT}"
        PID=\$(lsof -ti ":\$PORT" 2>/dev/null)
        if [[ -n "\$PID" ]]; then
            echo "🛑 Stopping proxy running on port \$PORT (PID: \$PID)..."
            kill "\$PID"
        else
            echo "⚠️ No proxy running on port \$PORT"
        fi
        ;;
    status)
        PORT="\${2:-\$PORT}"
        PID=\$(lsof -ti ":\$PORT" 2>/dev/null)
        if [[ -n "\$PID" ]]; then
            echo "✅ Proxy is running on port \$PORT (PID: \$PID)"
        else
            echo "❌ Proxy is not running on port \$PORT"
        fi
        ;;
    setup-env)
        if [[ ! -f "\$INSTALL_DIR/.env" ]]; then
            cp "\$INSTALL_DIR/.env.example" "\$INSTALL_DIR/.env"
            echo "📝 Created .env file from template."
            echo "⚠️ Please edit \$INSTALL_DIR/.env and set your configuration."
        else
            echo "ℹ️ .env file already exists."
        fi
        ;;
    update)
        echo "🔄 Updating claude-proxy..."
        # Re-run the installer script
        curl -s https://raw.githubusercontent.com/your-company/claude-code-proxy-installer/main/install.sh | bash
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
        echo "  setup-env       Create initial .env file from template"
        echo "  update          Update to the latest version"
        echo ""
        echo "Examples:"
        echo "  claude-proxy start"
        echo "  claude-proxy start 8083"
        echo "  claude-proxy stop"
        echo ""
        echo "Installed in: \$INSTALL_DIR"
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

# Handle arguments
if [[ "\$1" == "--gemini" || "\$1" == "-g" ]]; then
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
        sleep 2  # Wait for proxy to start
    fi

    # Run Claude with proxy
    echo "🔄 Running Claude with Gemini proxy (port: \$PORT)..."
    ANTHROPIC_BASE_URL="http://localhost:\$PORT" claude "\$@"
elif [[ "\$1" == "--help" || "\$1" == "-h" ]]; then
    echo "Claude Code with provider switching"
    echo ""
    echo "Usage: claudex [OPTIONS] [CLAUDE_ARGS]"
    echo ""
    echo "Options:"
    echo "  --gemini, -g [PORT]   Use Gemini proxy (default port: 8082)"
    echo "  --claude, -c          Use native Claude API (default)"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  claudex               # Use native Claude API"
    echo "  claudex -g            # Use Gemini proxy on port 8082"
    echo "  claudex -g 8083       # Use Gemini proxy on port 8083"
    echo "  claudex -c            # Use native Claude API explicitly"
    echo ""
    echo "Note: Additional arguments are passed to Claude Code"
elif [[ "\$1" == "--claude" || "\$1" == "-c" ]]; then
    # Run Claude with native API
    shift 1
    echo "🧠 Running Claude with native API..."
    unset ANTHROPIC_BASE_URL
    claude "\$@"
else
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
        cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
        print_info "Created .env file from template."
        print_warning "Please edit $INSTALL_DIR/.env and set your configuration."
    else
        print_info ".env file already exists."
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
