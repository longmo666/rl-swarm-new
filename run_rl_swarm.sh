#!/bin/bash

#General args
ROOT=$PWD

export PUB_MULTI_ADDRS
export PEER_MULTI_ADDRS
export HOST_MULTI_ADDRS
export IDENTITY_PATH

# Mac M-series optimizations
if [[ $(uname -m) == 'arm64' && $(uname) == 'Darwin' ]]; then
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export USE_MPS=1
  export CPU_ONLY=1
  echo "Applying Mac M-series optimizations"
fi

# Error handling function
handle_error() {
  local exit_code=$?
  echo "======================================"
  echo "RL Swarm encountered an error (code $exit_code)"

  if [[ $exit_code -eq 137 ]]; then
    echo "ERROR: Out of memory - try reducing model size or increase RAM"
    echo "Mac users: Make sure PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 is set"
  elif grep -q "failed to connect to bootstrap peers" logs/latest.txt 2>/dev/null; then
    echo "ERROR: Connection to bootstrap peers failed"
    echo "Check your internet connection or try an alternative peer"
  fi

  echo "For troubleshooting help, please file an issue at:"
  echo "https://github.com/gensyn-ai/rl-swarm/issues"
  echo "======================================"
  exit $exit_code
}

#Check if public multi-address is given else set to default
DEFAULT_PUB_MULTI_ADDRS=""
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

#Check if peer multi-address is given else set to default
DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ" # gensyn coordinator node

# Backup peer connection if primary fails
if command -v nc &> /dev/null; then
  if ! nc -z -w3 38.101.215.13 30002 &>/dev/null; then
    echo "Primary peer unreachable, using backup peer"
    DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.12/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ"
  fi
fi

PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

#Check if host multi-address is given else set to default
DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

# Path to an RSA private key. No need to specify if you
# just want a random Peer ID for this run.
DEFAULT_IDENTITY_PATH=""
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

#lets go!
echo "Getting requirements..."
pip install -r "$ROOT"/requirements-hivemind.txt
pip install -r "$ROOT"/requirements.txt
pip install "protobuf<5.28.0" --force-reinstall

if ! which nvidia-smi; then
   #You don't have a NVIDIA GPU
   CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
elif [ -n "$CPU_ONLY" ]; then
   # ... or we don't want to use it
   CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
else
   #NVIDIA GPU found
   pip install -r "$ROOT"/requirements_gpu.txt
   CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
fi

echo ">> Done!"
echo "Good luck in the swarm!"

# Set up error trapping and logging
trap handle_error ERR
mkdir -p logs
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).txt"
ln -sf "$LOG_FILE" logs/latest.txt

# Run the training
python -m hivemind_exp.gsm8k.train_single_gpu --identity_path "$IDENTITY_PATH" --public_maddr "$PUB_MULTI_ADDRS" --initial_peer "$PEER_MULTI_ADDRS" --host_maddr "$HOST_MULTI_ADDRS" --config "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"