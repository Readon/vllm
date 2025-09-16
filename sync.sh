#!/bin/bash

# sync.sh - Simplified synchronization script for vLLM development using rsync
# This script synchronizes Python files between the source code directory and the conda vllm environment installation
#
# Features:
# - Efficient file synchronization using rsync
# - Python file filtering with _version.py exclusion
# - Incremental and full sync modes
# - Dry-run capability
# - Comprehensive error handling
# - Detailed logging and reporting

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# =============================================================================
# Configuration Section
# =============================================================================

# Default configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR}"
readonly CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm}"
readonly BACKUP_DIR="${WORKSPACE_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${WORKSPACE_DIR}/sync_$(date +%Y%m%d_%H%M%S).log"

# Sync configuration
SYNC_MODE="${SYNC_MODE:-incremental}"  # full, incremental
SYNC_DIRECTION="${SYNC_DIRECTION:-source-to-target}"  # source-to-target, target-to-source, bidirectional
readonly MAX_RETRIES="${MAX_RETRIES:-3}"

# File patterns to include/exclude
readonly INCLUDE_PATTERNS_DEFAULT=("*.py" "*.json")
readonly EXCLUDE_PATTERNS_DEFAULT=("_version.py" "__pycache__" "*.pyc" ".git" "*.o" "*.so")

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Global variables
declare -a RSYNC_OUTPUT=()
declare -i DRY_RUN=0
declare -i VERBOSE=0
declare -g SYNC_STATS_DELETED=0
declare -g CONDA_PREFIX=""
declare -g VLLM_PACKAGE_DIR=""
declare -g SYNC_STATS_UPDATED=0
declare -g SYNC_STATS_CREATED=0
declare -g SYNC_STATS_SKIPPED=0
declare -g SYNC_STATS_FAILED=0
declare -a INCLUDE_PATTERNS=()
declare -a EXCLUDE_PATTERNS=()

# =============================================================================
# Logging Functions
# =============================================================================

log_init() {
    exec > >(tee -a "$LOG_FILE") 2>&1
    log_info "=== Sync session started at $(date) ==="
    log_info "Workspace directory: $WORKSPACE_DIR"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_debug() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Synchronizes Python files between the source code directory and the conda vllm environment installation using rsync.

OPTIONS:
    -h, --help              Show this help message
    -n, --dry-run           Show what would be done without actually doing it
    -v, --verbose           Enable verbose output
    --sync-mode MODE       Sync mode: full, incremental (default: incremental)
    --sync-direction DIR   Sync direction: source-to-target, target-to-source, bidirectional (default: source-to-target)
    --conda-env NAME       Conda environment name (default: vllm)
    --include PATTERNS     Comma-separated list of file patterns to include (default: *.py)
    --exclude PATTERNS     Comma-separated list of file patterns to exclude (default: _version.py,__pycache__,*.pyc,.git,*.o,*.so)

EXAMPLES:
    $SCRIPT_NAME                           # Standard incremental sync from source to target
    $SCRIPT_NAME --dry-run                # Preview changes without executing
    $SCRIPT_NAME --sync-mode full         # Full synchronization
    $SCRIPT_NAME --sync-direction target-to-source  # Sync from conda environment to source
    $SCRIPT_NAME --sync-direction bidirectional   # Bidirectional synchronization
    $SCRIPT_NAME --include "*.py"         # Only sync Python files (default)
    $SCRIPT_NAME --exclude "_version.py"  # Exclude _version.py files (default)

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -n|--dry-run)
                DRY_RUN=1
                shift
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            --sync-mode)
                SYNC_MODE="$2"
                shift 2
                ;;
            --sync-direction)
                SYNC_DIRECTION="$2"
                shift 2
                ;;
            --conda-env)
                CONDA_ENV_NAME="$2"
                shift 2
                ;;
            --include)
                IFS=',' read -ra INCLUDE_PATTERNS <<< "$2"
                shift 2
                ;;
            --exclude)
                IFS=',' read -ra EXCLUDE_PATTERNS <<< "$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Validate arguments
    if [[ ! "$SYNC_MODE" =~ ^(full|incremental)$ ]]; then
        log_error "Invalid sync mode: $SYNC_MODE. Must be 'full' or 'incremental'"
        exit 1
    fi

    if [[ ! "$SYNC_DIRECTION" =~ ^(source-to-target|target-to-source|bidirectional)$ ]]; then
        log_error "Invalid sync direction: $SYNC_DIRECTION. Must be 'source-to-target', 'target-to-source' or 'bidirectional'"
        exit 1
    fi
}

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if rsync is available
    if ! command -v rsync &> /dev/null; then
        log_error "rsync is required but not found. Please install rsync."
        exit 1
    fi
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        log_error "conda is required but not found. Please install conda."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# =============================================================================
# Environment Detection
# =============================================================================

detect_conda_environment() {
    log_info "Detecting conda environment..."
    
    # Check if the specified conda environment exists
    if ! conda info --envs | grep -q "^$CONDA_ENV_NAME "; then
        log_error "Conda environment '$CONDA_ENV_NAME' not found."
        exit 1
    fi
    
    # Get the conda prefix
    CONDA_PREFIX=$(conda info --envs | grep "^$CONDA_ENV_NAME " | awk '{print $NF}')
    log_success "Found conda environment: $CONDA_ENV_NAME at $CONDA_PREFIX"
}

find_vllm_installation() {
    log_info "Locating vLLM installation in conda environment..."
    
    # Find the vLLM package directory in the conda environment
    local python_version
    python_version=$(ls "$CONDA_PREFIX/lib/" | grep "^python3\." | sort -V | tail -n1)
    
    if [[ -z "$python_version" ]]; then
        log_error "Python installation not found in conda environment."
        exit 1
    fi
    
    VLLM_PACKAGE_DIR="$CONDA_PREFIX/lib/$python_version/site-packages/vllm"
    
    # Check if the vLLM package directory exists
    if [[ ! -d "$VLLM_PACKAGE_DIR" ]]; then
        log_error "vLLM package not found in the conda environment at $VLLM_PACKAGE_DIR"
        exit 1
    fi
    
    log_success "Found vLLM package at: $VLLM_PACKAGE_DIR"
}

# =============================================================================
# rsync Functions
# =============================================================================

build_rsync_exclude_args() {
    local exclude_args=()
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        exclude_args+=("--exclude=$pattern")
    done
    printf '%s\n' "${exclude_args[@]}"
}

build_rsync_include_args() {
    local include_args=()
    for pattern in "${INCLUDE_PATTERNS[@]}"; do
        include_args+=("--include=$pattern")
    done
    printf '%s\n' "${include_args[@]}"
}

perform_rsync_sync() {
    local source_dir="$1"
    local target_dir="$2"
    local direction="$3"
    local dry_run_flag=""
    local verbose_flag=""
    local delete_flag=""
    
    # Set flags based on options
    if [[ $DRY_RUN -eq 1 ]]; then
        dry_run_flag="--dry-run"
    fi
    
    if [[ $VERBOSE -eq 1 ]]; then
        verbose_flag="--verbose"
    fi
    
    if [[ "$SYNC_MODE" == "full" ]]; then
        delete_flag="--delete"
    fi
    
    # Build include and exclude arguments
    local include_args
    readarray -t include_args < <(build_rsync_include_args)
    
    local exclude_args
    readarray -t exclude_args < <(build_rsync_exclude_args)
    
    # Normalize paths
    source_dir=$(realpath "$source_dir" 2>/dev/null || echo "$source_dir")
    target_dir=$(realpath "$target_dir" 2>/dev/null || echo "$target_dir")
    
    # Ensure source directory ends with a slash for proper rsync behavior
    if [[ "$source_dir" != */ ]]; then
        source_dir="$source_dir/"
    fi
    
    log_info "Starting rsync synchronization: $source_dir -> $target_dir"
    
    # Build rsync command
    local rsync_cmd_array=("rsync" "-a")
    
    # Add flags if they are not empty
    if [[ -n "$dry_run_flag" ]]; then
        rsync_cmd_array+=("$dry_run_flag")
    fi
    
    if [[ -n "$verbose_flag" ]]; then
        rsync_cmd_array+=("$verbose_flag")
    fi
    
    if [[ -n "$delete_flag" ]]; then
        rsync_cmd_array+=("$delete_flag")
    fi
    
    # Add filtering arguments
    if [[ ${#include_args[@]} -gt 0 ]]; then
        rsync_cmd_array+=("${include_args[@]}")
    fi
    
    if [[ ${#exclude_args[@]} -gt 0 ]]; then
        rsync_cmd_array+=("${exclude_args[@]}")
    fi
    
    # Add default include/exclude rules
    rsync_cmd_array+=("--include=*/" "--exclude=*")
    
    # Add stats and paths
    rsync_cmd_array+=("--stats" "$source_dir" "$target_dir")
    
    # Convert array to string for logging
    local rsync_cmd="${rsync_cmd_array[*]}"
    
    if [[ $VERBOSE -eq 1 ]]; then
        log_debug "Executing: $rsync_cmd"
    fi
    
    # Capture rsync output for parsing
    local rsync_output
    if ! rsync_output=$("${rsync_cmd_array[@]}" 2>&1); then
        log_error "rsync command failed: $rsync_output"
        return 1
    fi
    
    # Store rsync output for later processing
    RSYNC_OUTPUT+=("$direction")
    RSYNC_OUTPUT+=("$rsync_output")
    
    # Parse rsync output for statistics
    parse_rsync_stats "$rsync_output" "$direction"
    
    # List modified files
    list_modified_files "$rsync_output" "$direction"
    
    log_success "rsync synchronization completed: $source_dir -> $target_dir"
    return 0
}

parse_rsync_stats() {
    local rsync_output="$1"
    local direction="$2"
    
    # Parse rsync statistics
    local created_files
    created_files=$(echo "$rsync_output" | grep -i "Number of created files:" | awk '{print $5}' || echo "0")
    
    local updated_files
    updated_files=$(echo "$rsync_output" | grep -i "Number of regular files transferred:" | awk '{print $6}' || echo "0")
    
    local deleted_files
    deleted_files=$(echo "$rsync_output" | grep -i "Number of deleted files:" | awk '{print $5}' || echo "0")
    
    # Ensure variables are numeric
    created_files=${created_files:-0}
    updated_files=${updated_files:-0}
    deleted_files=${deleted_files:-0}
    
    # Update global statistics based on direction
    case "$direction" in
        "source-to-target")
            ((SYNC_STATS_CREATED += created_files))
            ((SYNC_STATS_UPDATED += updated_files))
            ((SYNC_STATS_DELETED += deleted_files))
            ;;
        "target-to-source")
            # For reverse sync, we still count as created/updated/deleted from the perspective of the operation
            ((SYNC_STATS_CREATED += created_files))
            ((SYNC_STATS_UPDATED += updated_files))
            ((SYNC_STATS_DELETED += deleted_files))
            ;;
        "bidirectional-first")
            # First pass of bidirectional sync
            ((SYNC_STATS_CREATED += created_files))
            ((SYNC_STATS_UPDATED += updated_files))
            ((SYNC_STATS_DELETED += deleted_files))
            ;;
        "bidirectional-second")
            # Second pass of bidirectional sync
            ((SYNC_STATS_CREATED += created_files))
            ((SYNC_STATS_UPDATED += updated_files))
            ((SYNC_STATS_DELETED += deleted_files))
            ;;
    esac
    
    # Log detailed statistics if verbose
    if [[ $VERBOSE -eq 1 ]]; then
        log_info "rsync statistics for $direction:"
        log_info "  Created files: $created_files"
        log_info "  Updated files: $updated_files"
        log_info "  Deleted files: $deleted_files"
    fi
}

list_modified_files() {
    local rsync_output="$1"
    local direction="$2"
    
    log_info "Modified files in $direction sync:"
    
    # Extract file list from rsync output
    local file_count=0
    while IFS= read -r line; do
        # Skip empty lines and lines that don't look like file names
        if [[ -n "$line" ]] && [[ "$line" =~ \.(py)$ ]]; then
            echo "  $line"
            ((file_count++))
        fi
    done <<< "$(echo "$rsync_output" | grep -E "^[^/].*\.(py)$")"
    
    # If no specific files were listed, try a different approach
    if [[ $file_count -eq 0 ]]; then
        # Look for files in the verbose output
        while IFS= read -r line; do
            if [[ -n "$line" ]] && [[ "$line" =~ \.(py) ]]; then
                echo "  $line"
                ((file_count++))
            fi
        done <<< "$(echo "$rsync_output" | grep -E "^[^<>].*\.(py)")"
    fi
    
    # If still no files found, show a message
    if [[ $file_count -eq 0 ]]; then
        echo "  No Python files were modified in this sync operation."
    fi
}

# =============================================================================
# Synchronization Functions
# =============================================================================

execute_sync() {
    log_info "Starting synchronization..."
    
    local source_dir="$WORKSPACE_DIR/vllm"
    local target_dir="$VLLM_PACKAGE_DIR"
    local sync_success=1
    
    # Check if source directory exists
    if [[ ! -d "$source_dir" ]]; then
        log_error "Source directory does not exist: $source_dir"
        exit 1
    fi
    
    # Check if target directory exists
    if [[ ! -d "$target_dir" ]]; then
        log_error "Target directory does not exist: $target_dir"
        exit 1
    fi
    
    if [[ $DRY_RUN -eq 1 ]]; then
        log_info "DRY RUN MODE - No files will be actually modified"
    fi
    
    case "$SYNC_DIRECTION" in
        "source-to-target")
            if ! perform_rsync_sync "$source_dir" "$target_dir" "source-to-target"; then
                sync_success=0
            fi
            ;;
        "target-to-source")
            if ! perform_rsync_sync "$target_dir" "$source_dir" "target-to-source"; then
                sync_success=0
            fi
            ;;
        "bidirectional")
            log_info "Performing bidirectional synchronization..."
            
            # First pass: source to target
            log_info "First pass: source to target"
            if ! perform_rsync_sync "$source_dir" "$target_dir" "bidirectional-first"; then
                sync_success=0
            fi
            
            # Second pass: target to source
            log_info "Second pass: target to source"
            if ! perform_rsync_sync "$target_dir" "$source_dir" "bidirectional-second"; then
                sync_success=0
            fi
            ;;
    esac
    
    if [[ $sync_success -eq 1 ]]; then
        log_success "Synchronization completed"
    else
        log_error "Synchronization completed with errors"
        ((SYNC_STATS_FAILED++))
    fi
}

# =============================================================================
# Report Functions
# =============================================================================

generate_report() {
    log_info "Generating synchronization report..."
    
    local report_file="${WORKSPACE_DIR}/sync_report_$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "=== vLLM Synchronization Report ==="
        echo "Generated at: $(date)"
        echo "Workspace Directory: $WORKSPACE_DIR"
        echo "Conda Environment: $CONDA_ENV_NAME"
        echo "Target Directory: $VLLM_PACKAGE_DIR"
        echo "Sync Mode: $SYNC_MODE"
        echo "Sync Direction: $SYNC_DIRECTION"
        echo "Dry Run: $([[ $DRY_RUN -eq 1 ]] && echo "Yes" || echo "No")"
        echo ""
        
        echo "=== Statistics ==="
        echo "Files Updated: $SYNC_STATS_UPDATED"
        echo "Files Created: $SYNC_STATS_CREATED"
        echo "Files Deleted: $SYNC_STATS_DELETED"
        echo "Files Skipped: $SYNC_STATS_SKIPPED"
        echo "Files Failed: $SYNC_STATS_FAILED"
        echo ""
        
        # Add rsync output if available
        if [[ ${#RSYNC_OUTPUT[@]} -gt 0 ]]; then
            echo "=== rsync Output ==="
            local i=0
            while [[ $i -lt ${#RSYNC_OUTPUT[@]} ]]; do
                local direction="${RSYNC_OUTPUT[$i]}"
                local output="${RSYNC_OUTPUT[$((i+1))]}"
                echo ""
                echo "Sync direction: $direction"
                echo "$output"
                i=$((i+2))
            done
        fi
        
        echo ""
        echo "=== Log File ==="
        echo "Detailed log: $LOG_FILE"
        
    } > "$report_file"
    
    log_success "Report generated: $report_file"
}

show_summary() {
    echo ""
    echo "=== Synchronization Summary ==="
    echo "Files Updated: $SYNC_STATS_UPDATED"
    echo "Files Created: $SYNC_STATS_CREATED"
    echo "Files Deleted: $SYNC_STATS_DELETED"
    echo "Files Skipped: $SYNC_STATS_SKIPPED"
    echo "Files Failed: $SYNC_STATS_FAILED"
    echo ""
    
    # Show rsync output if verbose
    if [[ $VERBOSE -eq 1 ]] && [[ ${#RSYNC_OUTPUT[@]} -gt 0 ]]; then
        echo "=== rsync Output ==="
        local i=0
        while [[ $i -lt ${#RSYNC_OUTPUT[@]} ]]; do
            local direction="${RSYNC_OUTPUT[$i]}"
            local output="${RSYNC_OUTPUT[$((i+1))]}"
            echo ""
            echo "Sync direction: $direction"
            echo "$output"
            i=$((i+2))
        done
        echo ""
    fi
    
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[INFO] This was a dry run. No files were actually modified."
    fi
    
    echo "[SUCCESS] Synchronization completed successfully"
    echo "Detailed log: $LOG_FILE"
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Initialize logging
    log_init
    
    # Show configuration
    log_info "Configuration:"
    log_info "  Dry Run: $([[ $DRY_RUN -eq 1 ]] && echo "Yes" || echo "No")"
    log_info "  Verbose: $([[ $VERBOSE -eq 1 ]] && echo "Yes" || echo "No")"
    log_info "  Sync Mode: $SYNC_MODE"
    log_info "  Sync Direction: $SYNC_DIRECTION"
    
    # Check prerequisites
    check_prerequisites
    
    # Set default include/exclude patterns if not provided
    if [[ ${#INCLUDE_PATTERNS[@]} -eq 0 ]]; then
        INCLUDE_PATTERNS=("${INCLUDE_PATTERNS_DEFAULT[@]}")
    fi
    if [[ ${#EXCLUDE_PATTERNS[@]} -eq 0 ]]; then
        EXCLUDE_PATTERNS=("${EXCLUDE_PATTERNS_DEFAULT[@]}")
    fi
    
    # Detect and validate environment
    detect_conda_environment
    find_vllm_installation
    
    # Execute synchronization
    execute_sync
    
    # Generate report
    generate_report
    
    # Show summary
    show_summary
    
    log_info "=== Sync session completed at $(date) ==="
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Run main function with all arguments
main "$@"
