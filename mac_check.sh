#!/bin/bash
# Simple diagnostic script for Mac compatibility

echo "RL Swarm - Mac Compatibility Check"
echo "=================================="
echo "Machine type: $(uname -m)"

if [[ $(uname -m) == 'arm64' && $(uname) == 'Darwin' ]]; then
    echo "✅ Running on Apple Silicon (M-series)"

    # Check for critical environment variables
    if [[ -z "$PYTORCH_MPS_HIGH_WATERMARK_RATIO" ]]; then
        echo "⚠️  PYTORCH_MPS_HIGH_WATERMARK_RATIO not set (recommend setting to 0.0)"
    else
        echo "✅ PYTORCH_MPS_HIGH_WATERMARK_RATIO = $PYTORCH_MPS_HIGH_WATERMARK_RATIO"
    fi

    # Test peer connection
    if command -v nc &> /dev/null; then
        if nc -z -w3 38.101.215.13 30002 &>/dev/null; then
            echo "✅ Primary peer connection OK"
        else
            echo "⚠️  Primary peer unreachable, will try backup peer"
            if nc -z -w3 38.101.215.12 30002 &>/dev/null; then
                echo "✅ Backup peer connection OK"
            else
                echo "❌ Backup peer unreachable - check your internet connection"
            fi
        fi
    else
        echo "⚠️  'nc' command not found, can't check peer connectivity"
    fi

    # Check for Python and torch
    if command -v python3 &> /dev/null; then
        echo "✅ Python installed"
        if python3 -c "import torch; print(f'PyTorch: {torch.__version__}, MPS available: {torch.backends.mps.is_available()}')" 2>/dev/null; then
            echo "✅ PyTorch installed with MPS support"
        else
            echo "⚠️  PyTorch not installed or missing MPS support"
        fi
    else
        echo "❌ Python not found"
    fi
else
    echo "❌ Not running on Apple Silicon"
fi

echo "=================================="