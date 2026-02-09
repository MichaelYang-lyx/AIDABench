#!/bin/bash

# Auto Evaluation Runner Script

echo "Available Datasets:"
echo "1. all (Run chart, numerical, editing)"
echo "2. all_mini (Run chart_mini, numerical_mini, editing_mini)"
echo "3. chart"
echo "4. chart_mini"
echo "5. numerical"
echo "6. numerical_mini"
echo "7. editing"
echo "8. editing_mini"

if [ -z "$1" ]; then
    echo ""
    echo "Usage: ./run_eval.sh <dataset_name_or_option_number>"
    echo "Example: ./run_eval.sh numerical_mini"
    exit 1
fi

DATASET=$1

# Map numbers to names for convenience
case $1 in
    1) DATASET="all" ;;
    2) DATASET="all_mini" ;;
    3) DATASET="chart" ;;
    4) DATASET="chart_mini" ;;
    5) DATASET="numerical" ;;
    6) DATASET="numerical_mini" ;;
    7) DATASET="editing" ;;
    8) DATASET="editing_mini" ;;
esac

echo "Running evaluation for: $DATASET"
python3 auto_eval.py --dataset "$DATASET" --model_name "qwq-32b"
