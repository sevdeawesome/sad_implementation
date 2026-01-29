#!/bin/bash
# =============================================================================
# Judge Persona Steering Sweep
# =============================================================================
# Runs LLM judge evaluation across all 6 conditions (V1/V2/V3 × BASE/OPENCHAR)
# and generates comparison plots.
#
# Usage:
#   ./run_judge_sweep.sh              # Run all 6 conditions sequentially
#   ./run_judge_sweep.sh --parallel   # Run all 6 conditions in parallel
#   ./run_judge_sweep.sh --plots-only # Skip judging, just regenerate plots
#
# Output:
#   results/scripts/outputs/*.json    - Raw judge results
#   results/scripts/outputs/*.png     - Visualizations
# =============================================================================

set -e  # Exit on error

echo "Waiting 10 mins before running..."
# wait 10 mins before running
sleep 1000

echo "Running..."

# Configuration
PERSONAS="sarcasm goodness loving sycophancy poeticism"
STRENGTHS="-4 -2 -1.5 -1 -0.5 -0.25 0.25 0.5 1 1.5 2 4"
N_SAMPLES=14
OUTPUT_DIR="results/scripts/outputs"

# Parse args
PARALLEL=false
PLOTS_ONLY=false
for arg in "$@"; do
    case $arg in
        --parallel) PARALLEL=true ;;
        --plots-only) PLOTS_ONLY=true ;;
    esac
done

echo "============================================================"
echo "Judge Persona Steering Sweep"
echo "============================================================"
echo "Personas: $PERSONAS"
echo "Strengths: $STRENGTHS"
echo "Samples per comparison: $N_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Parallel mode: $PARALLEL"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# Function to run judge for one condition
run_judge() {
    local name=$1
    local results_dir=$2

    echo ""
    echo ">>> Running: $name"
    echo "    Results dir: $results_dir"

    python results/scripts/judge_persona_steering.py \
        --results-dir "$results_dir" \
        --output "$OUTPUT_DIR/${name}.png" \
        --personas $PERSONAS \
        --strengths $STRENGTHS \
        --n-samples $N_SAMPLES
}

if [ "$PLOTS_ONLY" = false ]; then
    echo ""
    echo "Starting LLM judge evaluations..."
    echo "(This will make many API calls - estimated cost ~\$25-30 for all 6 conditions)"
    echo ""

    if [ "$PARALLEL" = true ]; then
        echo "Running in PARALLEL mode (6 concurrent processes)"
        echo "Warning: This makes 6x concurrent API calls"
        echo ""

        run_judge "BASE_V1" "results/personalities_with_BASE_V1" &
        PID1=$!

        run_judge "BASE_V2" "results/personalities_with_BASE_MODEL_V2_vecs" &
        PID2=$!

        run_judge "BASE_V3" "results/personalities_with_BASE_MODEL_V3_vecs" &
        PID3=$!

        run_judge "OPENCHAR_V1" "results/personalities_with_OPENCHARACTER_V1" &
        PID4=$!

        run_judge "OPENCHAR_V2" "results/personalities_with_OPENCHARACTER_V2" &
        PID5=$!

        run_judge "OPENCHAR_V3" "results/personalities_with_OPENCHARACTER_V3_vecs" &
        PID6=$!

        echo "Waiting for all jobs to complete..."
        wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

    else
        echo "Running in SEQUENTIAL mode"
        echo ""

        # run_judge "BASE_V1" "results/personalities_with_BASE_V1"
        run_judge "BASE_V2" "results/personalities_with_BASE_MODEL_V2_vecs"
        # run_judge "BASE_V3" "results/personalities_with_BASE_MODEL_V3_vecs"
        # run_judge "OPENCHAR_V1" "results/personalities_with_OPENCHARACTER_V1"
        run_judge "OPENCHAR_V2" "results/personalities_with_OPENCHARACTER_V2"
        run_judge "OPENCHAR_V3" "results/personalities_with_OPENCHARACTER_V3_vecs"
    fi

    echo ""
    echo "============================================================"
    echo "LLM judge evaluations complete!"
    echo "============================================================"
fi

# Generate comparison plots
echo ""
echo "Generating comparison plots..."

# Double bar charts for each condition
python results/scripts/plot_judge_results.py --compare-all --persona sycophancy
python results/scripts/plot_judge_results.py --compare-all --persona sarcasm
python results/scripts/plot_judge_results.py --compare-all --persona humor
python results/scripts/plot_judge_results.py --compare-all --persona goodness

echo ""
echo "============================================================"
echo "All done!"
echo "============================================================"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.json 2>/dev/null | tail -20
echo ""
echo "Key plots to check:"
echo "  - $OUTPUT_DIR/sycophancy_comparison_grid.png (2x2 comparison)"
echo "  - $OUTPUT_DIR/effect_direction_summary.png (overview heatmap)"
echo "  - $OUTPUT_DIR/*_double_bars.png (detailed per-condition)"
