#!/bin/bash
set -e

# ============================================================
# Multi-label Gas Classification Experiment
# ============================================================
# Train on pure gases, test on mixtures to see if the model
# can identify component gases in gas mixtures.
# ============================================================

# Activate conda environment
# conda activate enose

# Proxy settings (if needed)
# export https_proxy=http://127.0.0.1:33210 http_proxy=http://127.0.0.1:33210 all_proxy=socks5://127.0.0.1:33211

MODE="${1:-train}"
ENCODER="${2:-tcn}"

# ============================================================
# Common parameters
# ============================================================
COMMON_ARGS=(
    --root .cache
    --max-length 512
    --num-channels 6
    --batch-size 32
    --max-epochs 200
    --learning-rate 1e-3
    --early-stopping-patience 30
    --accelerator gpu
    --precision bf16-mixed
    --num-workers 4
    --seed 42
)

# ============================================================
# Encoder-specific parameters
# ============================================================
TCN_ARGS=(
    --encoder tcn
    --hidden-dim 128
    --num-layers 4
    --dropout 0.1
)

MLP_ARGS=(
    --encoder mlp
    --hidden-dim 256
    --num-layers 3
    --dropout 0.1
)

# ============================================================
# Data sources
# ============================================================
# Training: Pure gases (Ethylene, Methane, CO from TwinGasSensorArrays)
TRAIN_SOURCES=(
    --train-sources twin_gas_pure
)

# Testing: Mixtures (Ethylene+Methane or Ethylene+CO)
TEST_SOURCES=(
    --test-sources gas_sensor_turbulent gas_sensor_dynamic
)

# ============================================================
# Functions
# ============================================================
train_model() {
    local encoder="$1"
    
    echo "=========================================="
    echo "Training Multi-label Classifier"
    echo "Encoder: $encoder"
    echo "=========================================="
    
    local encoder_args=()
    case "$encoder" in
        tcn)
            encoder_args=("${TCN_ARGS[@]}")
            ;;
        mlp)
            encoder_args=("${MLP_ARGS[@]}")
            ;;
        *)
            echo "Unknown encoder: $encoder. Use: tcn, mlp"
            exit 1
            ;;
    esac
    
    python experiment/multi_label/train.py \
        "${COMMON_ARGS[@]}" \
        "${encoder_args[@]}" \
        "${TRAIN_SOURCES[@]}" \
        "${TEST_SOURCES[@]}" \
        --experiment-name multi_label \
        --run-name "${encoder}_baseline"
}

show_info() {
    echo "=========================================="
    echo "Experiment Design"
    echo "=========================================="
    echo ""
    echo "Training data: TwinGasSensorArrays (pure gases)"
    echo "  - Ethylene (Ey): 10 concentration levels"
    echo "  - Methane (Me):  10 concentration levels"  
    echo "  - CO:            10 concentration levels"
    echo ""
    echo "Test data: Mixture datasets"
    echo "  - GasSensorTurbulent: Ethylene + Methane or Ethylene + CO"
    echo "  - GasSensorDynamic:   Ethylene + CO or Ethylene + Methane"
    echo ""
    echo "Task: Multi-label classification (3 classes)"
    echo "  - Can model trained on pure gases identify components in mixtures?"
    echo ""
}

# ============================================================
# Main
# ============================================================
case "$MODE" in
    train)
        train_model "$ENCODER"
        ;;
    info)
        show_info
        ;;
    help|*)
        echo "============================================================"
        echo "Multi-label Gas Classification Experiment"
        echo "============================================================"
        echo ""
        echo "Usage: $0 [MODE] [ENCODER]"
        echo ""
        echo "MODE:"
        echo "  train  - Train model (default)"
        echo "  info   - Show experiment design info"
        echo ""
        echo "ENCODER:"
        echo "  tcn   - Temporal Convolutional Network (default)"
        echo "  mlp   - Multi-layer Perceptron"
        echo ""
        echo "Examples:"
        echo "  $0 train tcn    # Train TCN"
        echo "  $0 train mlp    # Train MLP"
        echo "  $0 info         # Show experiment info"
        echo ""
        echo "Experiment design:"
        echo "  - Training: Pure gases (Ethylene, Methane, CO)"
        echo "  - Testing:  Mixtures (Ethylene+Methane, Ethylene+CO)"
        echo "  - Goal:     Can model identify components in mixtures?"
        exit 0
        ;;
esac

echo ""
echo "Done!"
