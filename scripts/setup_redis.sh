#!/bin/bash
set -e

# scripts/setup_redis.sh - Redis setup for decoder node pool coordination
# Usage: bash scripts/setup_redis.sh [--install] [--start] [--stop] [--status]
#
# The system uses Redis for:
#   - Decoder node pool discovery and registration
#   - Caching encoder outputs
#   - Distributed rate limiting
#   - Health check state
#
# Without Redis, the system falls back to disk-based storage.

REDIS_PORT="${UTS_REDIS_PORT:-6379}"
REDIS_CONTAINER_NAME="uts-redis"

detect_platform() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

check_redis_installed() {
    if command -v redis-server &>/dev/null; then
        local ver=$(redis-server --version 2>/dev/null || echo "unknown")
        echo "installed ($ver)"
        return 0
    fi
    echo "not_installed"
    return 1
}

check_redis_running() {
    if command -v redis-cli &>/dev/null; then
        if redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
            return 0
        fi
    fi
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^$REDIS_CONTAINER_NAME$"; then
        return 0
    fi
    return 1
}

cmd_install() {
    local status=$(check_redis_installed)
    echo "Redis status: $status"

    if echo "$status" | grep -q "installed"; then
        echo "Redis is already installed."
        exit 0
    fi

    local platform=$(detect_platform)
    echo "Installing Redis via package manager..."

    case "$platform" in
        linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y -qq redis-server
            elif command -v yum &>/dev/null; then
                sudo yum install -y redis
            elif command -v apk &>/dev/null; then
                sudo apk add redis
            else
                echo "Unsupported package manager. Try Docker fallback:"
                echo "  docker run -d --name $REDIS_CONTAINER_NAME -p $REDIS_PORT:6379 redis:7-alpine"
                exit 1
            fi
            ;;
        macos)
            if command -v brew &>/dev/null; then
                brew install redis
            else
                echo "Homebrew not found. Install it or use Docker:"
                echo "  docker run -d --name $REDIS_CONTAINER_NAME -p $REDIS_PORT:6379 redis:7-alpine"
                exit 1
            fi
            ;;
        *)
            echo "Unsupported platform. Use the Docker fallback:"
            echo "  docker run -d --name $REDIS_CONTAINER_NAME -p $REDIS_PORT:6379 redis:7-alpine"
            exit 1
            ;;
    esac

    echo "Redis installed successfully."
}

cmd_start() {
    if check_redis_running; then
        echo "Redis is already running on port $REDIS_PORT."
        exit 0
    fi

    local platform=$(detect_platform)
    case "$platform" in
        linux)
            if command -v systemctl &>/dev/null; then
                sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null
            elif command -v service &>/dev/null; then
                sudo service redis-server start 2>/dev/null || sudo service redis start 2>/dev/null
            else
                redis-server --daemonize yes
            fi
            ;;
        macos)
            if command -v brew &>/dev/null; then
                brew services start redis
            else
                redis-server --daemonize yes
            fi
            ;;
        *)
            echo "Starting via Docker..."
            docker run -d --name "$REDIS_CONTAINER_NAME" -p "$REDIS_PORT":6379 redis:7-alpine
            ;;
    esac

    if check_redis_running; then
        echo "Redis started on port $REDIS_PORT."
    else
        echo "Failed to start Redis. Try the Docker fallback:"
        echo "  docker run -d --name $REDIS_CONTAINER_NAME -p $REDIS_PORT:6379 redis:7-alpine"
        exit 1
    fi
}

cmd_stop() {
    local platform=$(detect_platform)
    case "$platform" in
        linux)
            if command -v systemctl &>/dev/null; then
                sudo systemctl stop redis-server 2>/dev/null || sudo systemctl stop redis 2>/dev/null || true
            elif command -v service &>/dev/null; then
                sudo service redis-server stop 2>/dev/null || sudo service redis stop 2>/dev/null || true
            fi
            ;;
        macos)
            if command -v brew &>/dev/null; then
                brew services stop redis 2>/dev/null || true
            fi
            ;;
    esac

    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^$REDIS_CONTAINER_NAME$"; then
        docker rm -f "$REDIS_CONTAINER_NAME" 2>/dev/null || true
    fi

    echo "Redis stopped."
}

cmd_status() {
    echo "=== Redis Status ==="
    echo ""

    local installed=$(check_redis_installed)
    echo "Installed: $installed"

    if check_redis_running; then
        echo "Running:  yes"
        if command -v redis-cli &>/dev/null; then
            echo "Version:  $(redis-cli -p "$REDIS_PORT" INFO SERVER 2>/dev/null | grep 'redis_version' | cut -d: -f2 || echo 'unknown')"
            echo "Uptime:   $(redis-cli -p "$REDIS_PORT" INFO SERVER 2>/dev/null | grep 'uptime_in_seconds' | cut -d: -f2 || echo 'unknown') seconds"
            echo "Clients:  $(redis-cli -p "$REDIS_PORT" INFO CLIENTS 2>/dev/null | grep 'connected_clients' | cut -d: -f2 || echo 'unknown')"
            echo "Memory:   $(redis-cli -p "$REDIS_PORT" INFO MEMORY 2>/dev/null | grep 'used_memory_human' | cut -d: -f2 || echo 'unknown')"
        fi
    else
        echo "Running:  no"
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^$REDIS_CONTAINER_NAME$"; then
            echo "Note: Docker container '$REDIS_CONTAINER_NAME' exists but may not be running."
        fi
    fi

    echo ""
    echo "=== Configuration ==="
    echo ""
    echo "Set the UTS_REDIS_URL environment variable in your shell or .env file:"
    echo ""
    echo '  export UTS_REDIS_URL="redis://localhost:6379/0"'
    echo ""
    echo "Or with authentication:"
    echo ""
    echo '  export UTS_REDIS_URL="redis://:password@localhost:6379/0"'
    echo ""
    echo "The RedisManager (utils/redis_manager.py) also reads REDIS_URL directly."
    echo "Additional env vars:"
    echo "  REDIS_KEY_PREFIX        Key prefix for all entries (default: translation:)"
    echo "  REDIS_CONN_TIMEOUT      Connection timeout in seconds (default: 2)"
    echo "  REDIS_READ_TIMEOUT      Read timeout in seconds (default: 2)"
    echo "  REDIS_HEALTH_CHECK_INT  Health check interval in seconds (default: 30)"
    echo "  REDIS_USE_MSGPACK       Use MessagePack serialization (default: true)"
    echo ""
    echo "Docker fallback (if Redis is not installed):"
    echo ""
    echo "  docker run -d --name $REDIS_CONTAINER_NAME -p $REDIS_PORT:6379 redis:7-alpine"
}

case "${1:-}" in
    --install) cmd_install ;;
    --start)   cmd_start ;;
    --stop)    cmd_stop ;;
    --status)  cmd_status ;;
    *)
        cat <<EOF
Usage: bash scripts/setup_redis.sh [--install] [--start] [--stop] [--status]

Options:
  --install   Install Redis via system package manager (apt/brew/yum)
  --start     Start the Redis server
  --stop      Stop the Redis server
  --status    Print connection status and configuration guide

Without Redis, the system falls back to disk-based storage.
EOF
        exit 1
        ;;
esac
