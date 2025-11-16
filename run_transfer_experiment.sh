#!/bin/bash

# Transfer Quantization Experiment Runner
# This script helps you run the transfer quantization experiment with proper setup

set -e  # Exit on error

echo "=================================="
echo "Transfer Quantization Experiment"
echo "=================================="
echo ""

# Check if config file is provided
CONFIG_FILE=${1:-"transfer_config.json"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo ""
    echo "Usage: ./run_transfer_experiment.sh [config_file]"
    echo ""
    echo "Available configs:"
    echo "  - transfer_config.json (default)"
    echo ""
    echo "Or create your own config based on the template."
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python -c "import torch; import transformers; import datasets" 2>/dev/null || {
    echo "Error: Missing dependencies!"
    echo "Install with: pip install torch transformers datasets tqdm"
    exit 1
}
echo "✓ All dependencies installed"
echo ""

# Run the experiment
echo "Starting experiment..."
echo "This may take 30-60 minutes depending on your hardware."
echo ""

python transfer_quantization_experiment.py "$CONFIG_FILE"

echo ""
echo "=================================="
echo "Experiment Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - transfer_experiment_results.json"
echo "  - scaling_factors_350m.json"
echo ""
echo "See TRANSFER_EXPERIMENT_README.md for analysis instructions."
