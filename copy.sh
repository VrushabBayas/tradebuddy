#!/bin/bash

# FinGPT Model Copy Script
# This script copies FinGPT models into a Docker container and creates necessary directory structure

set -e # Exit on any error

echo "🚀 Starting FinGPT Model Copy Process..."
echo "======================================"

# Function to check if Docker is running
check_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
  fi
  echo "✅ Docker is running"
}

# Function to check if container exists
check_container() {
  if ! docker ps -a --format "table {{.Names}}" | grep -q "tradebuddy-fingpt"; then
    echo "❌ Error: Container 'tradebuddy-fingpt' not found."
    echo "Please make sure the container is created first."
    exit 1
  fi
  echo "✅ Container 'tradebuddy-fingpt' found"
}

# Function to check if source directories exist
check_source_dirs() {
  local missing_dirs=()

  if [ ! -d "~/fingpt-models/fingpt-mt_llama3-8b_lora/" ]; then
    missing_dirs+=("~/fingpt-models/fingpt-mt_llama3-8b_lora/")
  fi

  if [ ! -d "~/fingpt-models/fingpt-sentiment_llama2-13b_lora/" ]; then
    missing_dirs+=("~/fingpt-models/fingpt-sentiment_llama2-13b_lora/")
  fi

  if [ ${#missing_dirs[@]} -gt 0 ]; then
    echo "❌ Error: The following source directories are missing:"
    for dir in "${missing_dirs[@]}"; do
      echo "   - $dir"
    done
    echo "Please download the models first."
    exit 1
  fi

  echo "✅ Source model directories found"
}

# Function to create directory structure in container
create_directories() {
  echo "📁 Creating directory structure in container..."

  # Create directory for LLama3 8B model
  echo "   Creating directory for fingpt-mt_llama3-8b_lora..."
  docker exec tradebuddy-fingpt sh -c "mkdir -p /app/model_cache/models--FinGPT--fingpt-mt_llama3-8b_lora/snapshots/main"

  # Create directory for LLama2 13B sentiment model
  echo "   Creating directory for fingpt-sentiment_llama2-13b_lora..."
  docker exec tradebuddy-fingpt sh -c "mkdir -p /app/model_cache/models--FinGPT--fingpt-sentiment_llama2-13b_lora/snapshots/main"

  echo "✅ Directory structure created successfully"
}

# Function to copy models
copy_models() {
  echo "📦 Copying models to container (this may take a while - ~32GB)..."

  # Copy the LLama3 8B model
  echo "   Copying fingpt-mt_llama3-8b_lora model..."
  docker cp ~/fingpt-models/fingpt-mt_llama3-8b_lora/. tradebuddy-fingpt:/app/model_cache/models--FinGPT--fingpt-mt_llama3-8b_lora/snapshots/main/

  # Copy the LLama2 13B sentiment model
  echo "   Copying fingpt-sentiment_llama2-13b_lora model..."
  docker cp ~/fingpt-models/fingpt-sentiment_llama2-13b_lora/. tradebuddy-fingpt:/app/model_cache/models--FinGPT--fingpt-sentiment_llama2-13b_lora/snapshots/main/

  echo "✅ Models copied successfully"
}

# Function to verify the copy operation
verify_copy() {
  echo "🔍 Verifying copied models..."

  # Check if models exist in container
  echo "   Checking fingpt-mt_llama3-8b_lora..."
  if docker exec tradebuddy-fingpt sh -c "[ -d '/app/model_cache/models--FinGPT--fingpt-mt_llama3-8b_lora/snapshots/main' ]"; then
    echo "   ✅ fingpt-mt_llama3-8b_lora directory exists"
  else
    echo "   ❌ fingpt-mt_llama3-8b_lora directory not found"
  fi

  echo "   Checking fingpt-sentiment_llama2-13b_lora..."
  if docker exec tradebuddy-fingpt sh -c "[ -d '/app/model_cache/models--FinGPT--fingpt-sentiment_llama2-13b_lora/snapshots/main' ]"; then
    echo "   ✅ fingpt-sentiment_llama2-13b_lora directory exists"
  else
    echo "   ❌ fingpt-sentiment_llama2-13b_lora directory not found"
  fi

  # Show disk usage
  echo "   Container disk usage:"
  docker exec tradebuddy-fingpt sh -c "du -sh /app/model_cache/models--*"
}

# Function to restart container
restart_container() {
  echo "🔄 Restarting container to test model loading..."
  docker restart tradebuddy-fingpt

  # Wait a moment for container to start
  sleep 5

  if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "tradebuddy-fingpt.*Up"; then
    echo "✅ Container restarted successfully"
  else
    echo "❌ Container failed to restart properly"
    exit 1
  fi
}

# Main execution
main() {
  echo "Starting FinGPT model copy process..."
  echo "This script will:"
  echo "1. Check Docker and container status"
  echo "2. Verify source model directories"
  echo "3. Create directory structure in container"
  echo "4. Copy models (~32GB of data)"
  echo "5. Restart container for testing"
  echo ""

  read -p "Do you want to continue? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
  fi

  check_docker
  check_container
  check_source_dirs
  create_directories
  copy_models
  verify_copy
  restart_container

  echo ""
  echo "🎉 FinGPT model copy process completed successfully!"
  echo "======================================================"
  echo "Your models are now available in the Docker container."
  echo "You can now test the model loading functionality."
}

# Run the main function
main "$@"
