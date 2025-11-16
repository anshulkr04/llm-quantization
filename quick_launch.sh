#!/bin/bash

# Quick Launcher for Transfer Quantization Experiment
# Automatically selects appropriate config based on available memory

set -e

echo "========================================"
echo "Transfer Quantization Quick Launcher"
echo "========================================"
echo ""

# Check available memory (Linux)
if command -v free &> /dev/null; then
    available_mem=$(free -m | awk '/^Mem:/{print $7}')
    echo "Available memory: ${available_mem} MB"
    echo ""
    
    if [ "$available_mem" -lt 8000 ]; then
        config="transfer_config_tiny.json"
        echo "⚠️  Low memory detected (< 8GB available)"
        echo "   Using TINY config (GPT2 → GPT2-Medium)"
    elif [ "$available_mem" -lt 16000 ]; then
        config="transfer_config_small.json"
        echo "ℹ️  Moderate memory detected (8-16GB available)"
        echo "   Using SMALL config (OPT-350M → OPT-1.3B)"
    else
        config="transfer_config.json"
        echo "✓ Good memory detected (> 16GB available)"
        echo "   Using FULL config (OPT-350M → OPT-1.3B)"
    fi
else
    echo "Unable to detect memory, using TINY config for safety"
    config="transfer_config_tiny.json"
fi

echo ""
echo "Selected configuration: $config"
echo ""

# Allow user to override
read -p "Press ENTER to continue, or type a config filename to use instead: " user_config

if [ ! -z "$user_config" ]; then
    config="$user_config"
    echo "Using user-specified config: $config"
fi

echo ""
echo "========================================"
echo "Starting experiment with: $config"
echo "========================================"
echo ""

# Run the experiment
./run_transfer_experiment.sh "$config"
