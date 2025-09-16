#!/bin/bash

# run.sh - vLLM server startup script with Turing optimizations
# Launches vLLM server with Qwen3-30B model and optimized settings for decode performance

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  check_prerequisites  - Check system prerequisites only"
    echo "  setup_environment    - Setup Turing environment only"
    echo "  start_server         - Start vLLM server only"
    echo "  stop_server          - Stop running server"
    echo "  status               - Check server status"
    echo "  help                 - Show this help message"
    echo ""
    echo "If no command is provided, runs the full server startup sequence."
}

# Configuration
MODEL_PATH="/models/gpustack_cache/model_scope/tclf90/Qwen3-235B-A22B-Instruct-2507-AWQ/"
MODEL_NAME="qwen3-235b"
SERVER_PORT=8000
SERVER_HOST="0.0.0.0"
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.95
CONDA_ENV_NAME="vllm"
LOG_FILE="vllm_server.log"
PID_FILE="vllm_server.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # CRITICAL: Ensure we're not running from vllm source directory
    current_dir=$(basename "$(pwd)")
    if [ "$current_dir" = "vllm" ] && [ -f "setup.py" ] && [ -d "vllm" ]; then
        log_error "FORBIDDEN: Cannot run vLLM server from vllm source directory!"
        log_error "Current directory: $(pwd)"
        log_info "Please run this script from outside the vllm source directory."
        log_info "Example: cd /home/user && /path/to/vllm/run.sh"
        exit 1
    fi

    # Check if model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model path not found: $MODEL_PATH"
        log_info "Please ensure the model is downloaded and available at the specified path."
        exit 1
    fi

    log_success "Model path verified: $MODEL_PATH"
    
    # Check CUDA availability
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    # Check GPU count
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    log_info "Detected $gpu_count GPU(s)"
    
    if [ "$gpu_count" -lt "$TENSOR_PARALLEL_SIZE" ]; then
        log_warning "Requested tensor parallel size ($TENSOR_PARALLEL_SIZE) exceeds available GPUs ($gpu_count)"
        log_info "Adjusting tensor parallel size to $gpu_count"
        TENSOR_PARALLEL_SIZE=$gpu_count
    fi
    
    # Check conda environment
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install conda or miniconda."
        exit 1
    fi
    
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_error "Conda environment '${CONDA_ENV_NAME}' not found."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Function to setup environment variables for Turing optimizations
setup_turing_environment() {
    log_info "Setting up Turing optimization environment..."
    
    # Check if setup script exists and source it
    local env_script="./setup_turing_env.sh"
    if [ -f "$env_script" ]; then
        log_info "Sourcing Turing environment setup..."
        source "$env_script"
    else
        log_info "Setting Turing environment variables directly..."
        
        # Force Turing backend selection
        export VLLM_ATTENTION_BACKEND="TURING"
        
        # Use V0 for Turing backend compatibility
        export VLLM_USE_V1=0
        
        # Enable Triton legacy PTX assembler for Turing
        export TRITON_USE_LEGACY_PTX_ASSEMBLER=1
        
        # Optimize for Turing architecture
        export CUDA_ARCH_LIST="7.5"
        
        # Memory optimization settings
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
        
        # Additional optimizations for large context
        export VLLM_WORKER_MULTIPROC_METHOD="spawn"
        export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
        
        log_info "Turing environment variables set"
    fi
    
    # Validate Turing GPU availability (only if python3 and torch are available)
    if command -v python3 &> /dev/null; then
        python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        turing_count = 0
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            if capability == (7, 5):
                turing_count += 1
        
        if turing_count > 0:
            print(f'Found {turing_count} Turing GPU(s) - optimizations will be enabled')
        else:
            print('No Turing GPUs detected - standard attention backend will be used')
    else:
        print('CUDA not available')
except ImportError:
    print('Torch not available - skipping GPU validation')
" || {
        log_warning "GPU validation failed, continuing with standard settings"
    }
    else
        log_warning "python3 not available - skipping GPU validation"
    fi
}

# Function to check if server is already running
check_existing_server() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_warning "vLLM server is already running (PID: $pid)"
            log_info "To stop the existing server, run: kill $pid"
            log_info "Or use: ./stop_server.sh"
            exit 1
        else
            log_info "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi
    
    # Check if port is in use
    if netstat -tuln 2>/dev/null | grep -q ":$SERVER_PORT "; then
        log_error "Port $SERVER_PORT is already in use"
        log_info "Please stop the service using this port or choose a different port"
        exit 1
    fi
}

# Function to start vLLM server
start_vllm_server() {
    log_info "Starting vLLM server with Turing optimizations..."
    log_info "Configuration:"
    log_info "  Model: $MODEL_NAME ($MODEL_PATH)"
    log_info "  Server: $SERVER_HOST:$SERVER_PORT"
    log_info "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    log_info "  Max Model Length: $MAX_MODEL_LEN"
    log_info "  Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS"
    log_info "  Max Sequences: $MAX_NUM_SEQS"
    log_info "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    
    # Activate conda environment and ensure we use the installed vLLM package
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"

    # Verify we're using the conda environment's vLLM, not source directory
    log_info "Verifying vLLM installation..."
    python -c "import vllm; print(f'Using vLLM from: {vllm.__file__}')" || {
        log_error "Failed to import vLLM from conda environment"
        exit 1
    }

    # Start the server using vllm serve command
    vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --disable-log-stats \
        --enable-chunked-prefill \
        --disable-sliding-window \
        --dtype float16 \
        --kv-cache-dtype fp8_e5m2 \
        --swap-space 16 \
        --cpu-offload-gb 0 > "$LOG_FILE" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "$PID_FILE"
    
    log_success "vLLM server started with PID: $server_pid"
    log_info "Server logs: $LOG_FILE"
    log_info "PID file: $PID_FILE"
    
    conda deactivate
}

# Function to wait for server readiness
wait_for_server() {
    log_info "Waiting for server to be ready..."
    
    local max_attempts=120  # 10 minutes
    local attempt=1
    local server_url="http://${SERVER_HOST}:${SERVER_PORT}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "${server_url}/health" > /dev/null 2>&1; then
            log_success "Server is ready and responding!"
            log_info "Health endpoint: ${server_url}/health"
            log_info "OpenAI API endpoint: ${server_url}/v1"
            return 0
        fi
        
        # Check if server process is still running
        if [ -f "$PID_FILE" ]; then
            local pid=$(cat "$PID_FILE")
            if ! kill -0 "$pid" 2>/dev/null; then
                log_error "Server process died unexpectedly"
                log_info "Check logs: $LOG_FILE"
                return 1
            fi
        fi
        
        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        fi
        
        sleep 5
        ((attempt++))
    done
    
    log_error "Server failed to start within expected time"
    log_info "Check logs: $LOG_FILE"
    return 1
}

# Function to display server information
display_server_info() {
    log_success "vLLM server is running successfully!"
    log_info "=" * 60
    log_info "SERVER INFORMATION"
    log_info "=" * 60
    log_info "Model: $MODEL_NAME"
    log_info "Base URL: http://${SERVER_HOST}:${SERVER_PORT}"
    log_info "Health Check: http://${SERVER_HOST}:${SERVER_PORT}/health"
    log_info "OpenAI API: http://${SERVER_HOST}:${SERVER_PORT}/v1"
    log_info "Docs: http://${SERVER_HOST}:${SERVER_PORT}/docs"
    log_info ""
    log_info "EXAMPLE USAGE:"
    log_info "curl -X POST http://${SERVER_HOST}:${SERVER_PORT}/v1/completions \\"
    log_info "  -H 'Content-Type: application/json' \\"
    log_info "  -d '{"
    log_info "    \"model\": \"$MODEL_NAME\","
    log_info "    \"prompt\": \"Hello, how are you?\","
    log_info "    \"max_tokens\": 100"
    log_info "  }'"
    log_info ""
    log_info "MANAGEMENT:"
    log_info "  Logs: tail -f $LOG_FILE"
    log_info "  Stop: kill \$(cat $PID_FILE)"
    log_info "  Status: kill -0 \$(cat $PID_FILE) && echo 'Running' || echo 'Stopped'"
}

# Function to create stop script
create_stop_script() {
    local stop_script="./stop_server.sh"
    
    cat > "$stop_script" << EOF
#!/bin/bash
# Auto-generated stop script for vLLM server

PID_FILE="$PID_FILE"
LOG_FILE="$LOG_FILE"

if [ -f "\$PID_FILE" ]; then
    PID=\$(cat "\$PID_FILE")
    echo "Stopping vLLM server (PID: \$PID)..."
    
    if kill -TERM "\$PID" 2>/dev/null; then
        echo "Sent SIGTERM to process \$PID"
        
        # Wait for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "\$PID" 2>/dev/null; then
                echo "Server stopped gracefully"
                rm -f "\$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "Forcing server shutdown..."
        kill -KILL "\$PID" 2>/dev/null
        rm -f "\$PID_FILE"
        echo "Server stopped forcefully"
    else
        echo "Process \$PID not found or already stopped"
        rm -f "\$PID_FILE"
    fi
else
    echo "PID file not found. Server may not be running."
fi
EOF
    
    chmod +x "$stop_script"
    log_info "Created stop script: $stop_script"
}

# Cleanup function
cleanup() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        log_info "Cleaning up server process (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Function to handle command line arguments
handle_command() {
    local command="${1:-full_start}"
    
    case "$command" in
        check_prerequisites)
            log_info "Checking prerequisites only..."
            check_prerequisites
            ;;
        setup_environment)
            log_info "Setting up Turing environment only..."
            setup_turing_environment
            ;;
        start_server)
            log_info "Starting vLLM server only..."
            check_existing_server
            start_vllm_server
            create_stop_script
            wait_for_server && display_server_info
            ;;
        stop_server)
            log_info "Stopping server..."
            cleanup
            ;;
        status)
            if [ -f "$PID_FILE" ]; then
                local pid=$(cat "$PID_FILE")
                if kill -0 "$pid" 2>/dev/null; then
                    log_success "Server is running (PID: $pid)"
                else
                    log_error "Server process not found (stale PID file)"
                    rm -f "$PID_FILE"
                fi
            else
                log_info "Server is not running"
            fi
            ;;
        help|--help|-h)
            usage
            exit 0
            ;;
        full_start)
            # Full startup sequence
            log_info "Starting vLLM server with Turing decode optimizations"
            log_info "=" * 60
            
            check_prerequisites
            setup_turing_environment
            check_existing_server
            start_vllm_server
            create_stop_script
            
            if wait_for_server; then
                display_server_info
                
                # Keep the script running to maintain the server
                log_info "Server is running. Press Ctrl+C to stop."
                
                # Monitor server process
                while true; do
                    if [ -f "$PID_FILE" ]; then
                        local pid=$(cat "$PID_FILE")
                        if ! kill -0 "$pid" 2>/dev/null; then
                            log_error "Server process died unexpectedly"
                            log_info "Check logs: $LOG_FILE"
                            exit 1
                        fi
                    else
                        log_error "PID file disappeared"
                        exit 1
                    fi
                    
                    sleep 10
                done
            else
                log_error "Failed to start server"
                exit 1
            fi
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Main execution
main() {
    local command="${1:-full_start}"
    handle_command "$command"
}

# Run main function
main "$@"
