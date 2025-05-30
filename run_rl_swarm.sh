#!/bin/bash

set -euo pipefail

# General arguments
ROOT=$PWD

export PUB_MULTI_ADDRS
export PEER_MULTI_ADDRS
export HOST_MULTI_ADDRS
export IDENTITY_PATH
export CONNECT_TO_TESTNET
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120  # 2 minutes

# Default multi-addresses
DEFAULT_PUB_MULTI_ADDRS=""
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.14/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ"
PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

# Identity path
DEFAULT_IDENTITY_PATH="$ROOT/swarm.pem"
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

SMALL_SWARM_CONTRACT="0x69C6e1D608ec64885E7b185d39b04B491a71768C"
BIG_SWARM_CONTRACT="0x6947c6E196a48B77eFa9331EC1E3e45f3Ee5Fd58"

# 强制使用CPU模式
CPU_ONLY=true
CONNECT_TO_TESTNET=true   # 默认连接 Testnet
USE_BIG_SWARM=false       # 默认小 swarm (Math A)
PARAM_B=0.5               # 默认 0.5b
PUSH_TO_HF=false          # 默认不推送到 Hugging Face Hub
HUGGINGFACE_ACCESS_TOKEN="None"
HF_TOKEN=${HF_TOKEN:-""} # 默认空 HF_TOKEN

GREEN_TEXT="\033[32m"
BLUE_TEXT="\033[34m"
RESET_TEXT="\033[0m"

echo_green() {
  echo -e "${GREEN_TEXT}$1${RESET_TEXT}"
}

echo_blue() {
  echo -e "${BLUE_TEXT}$1${RESET_TEXT}"
}

ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

cleanup() {
  echo_green ">> Shutting down trainer..."
  rm -r $ROOT_DIR/modal-login/temp-data/*.json 2> /dev/null || true
  kill -- -$$ || true
  exit 0
}
trap cleanup EXIT

# Apply defaults without prompting
if [ "$USE_BIG_SWARM" = true ]; then
  SWARM_CONTRACT="$BIG_SWARM_CONTRACT"
else
  SWARM_CONTRACT="$SMALL_SWARM_CONTRACT"
fi

# If connecting to testnet, run modal-login and prepare environment
if [ "$CONNECT_TO_TESTNET" = true ]; then
  echo "Please login to create an Ethereum Server Wallet"
  cd modal-login
  # Remove npm lockfile to avoid Yarn warning
rm -f package-lock.json 2> /dev/null || true

# Install Node.js and Yarn if missing
  if ! command -v node > /dev/null 2>&1; then
      export NVM_DIR="$HOME/.nvm"
      if [ ! -d "$NVM_DIR" ]; then
          curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
      fi
      [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
      nvm install node
  fi
  if ! command -v yarn > /dev/null 2>&1; then
      if grep -qi "ubuntu" /etc/os-release 2> /dev/null || uname -r | grep -qi "microsoft"; then
          curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
          echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
          sudo apt update && sudo apt install -y yarn
      else
          npm install -g --silent yarn
      fi
  fi
  yarn install
  yarn dev > /dev/null 2>&1 &
  SERVER_PID=$!
  echo "Started server process: $SERVER_PID"
  sleep 5
  if open http://localhost:3000 2> /dev/null; then
      echo_green ">> Successfully opened http://localhost:3000 in your default browser."
  else
      echo ">> Failed to open http://localhost:3000. Please open it manually."
  fi
  cd ..

  echo_green ">> Waiting for modal userData.json to be created..."
  # Copy backup credentials
  cp ~/rl-swarm-backup/swarm.pem ~/rl-swarm/ && \
  cp ~/rl-swarm-backup/userApiKey.json ~/rl-swarm-backup/userData.json /root/rl-swarm/modal-login/temp-data/

  while [ ! -f "modal-login/temp-data/userData.json" ]; do
      sleep 5
  done
  echo "Found userData.json. Proceeding..."

  ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
  echo "Your ORG_ID is set to: $ORG_ID"
  echo "Waiting for API key to become activated..."
  while true; do
      STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
      if [[ "$STATUS" == "activated" ]]; then
          echo "API key is activated! Proceeding..."
          break
      else
          echo "Waiting for API key to be activated..."
          sleep 5
      fi
  done

  ENV_FILE="$ROOT/modal-login/.env"
  if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
  else
      sed -i "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
  fi
fi

echo_green ">> Getting requirements..."

pip install --upgrade pip
# 强制使用CPU requirements
echo_green ">> Installing CPU-only requirements..."
pip install -r "$ROOT/requirements-cpu.txt"

# 创建CPU配置文件（如果不存在）
CPU_CONFIG_PATH="$ROOT/grpo-qwen-2.5-0.5b-deepseek-r1-cpu.yaml"
if [ ! -f "$CPU_CONFIG_PATH" ]; then
    echo_green ">> Creating CPU configuration file..."
    cat > "$CPU_CONFIG_PATH" << 'EOF'
# Model arguments
model_revision: main
torch_dtype: float32
bf16: false
tf32: false

# Dataset arguments
dataset_id_or_path: 'openai/gsm8k'

# Training arguments
max_steps: 100 # Original 450
gradient_accumulation_steps: 2
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.03

# GRPO arguments
use_vllm: false
num_generations: 4
per_device_train_batch_size: 2
beta: 0.001
max_prompt_length: 256
max_completion_length: 512

# Logging arguments
logging_strategy: steps
logging_steps: 2
report_to:
- tensorboard
save_strategy: "steps"
save_steps: 25
seed: 42

# Script arguments
max_rounds: 10000

# Model-specific arguments
model_name_or_path: unsloth/Qwen2.5-0.5B-Instruct
output_dir: runs/gsm8k/multinode/Qwen2.5-0.5B-Instruct-Gensyn-Swarm
EOF
fi

CONFIG_PATH="$CPU_CONFIG_PATH"
GAME="gsm8k"

echo_green ">> Done!"

if [ -n "$HF_TOKEN" ]; then
  HUGGINGFACE_ACCESS_TOKEN=$HF_TOKEN
elif [ "$PUSH_TO_HF" = true ]; then
  HUGGINGFACE_ACCESS_TOKEN="$HUGGINGFACE_ACCESS_TOKEN"
else
  HUGGINGFACE_ACCESS_TOKEN="None"
fi

echo_green ">> Good luck in the swarm!"
echo_blue ">> Post about rl-swarm on X/twitter! --> https://tinyurl.com/swarmtweet"
echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"

# 设置环境变量强制使用CPU
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=""

echo_green ">> Using CPU configuration: $CONFIG_PATH"
echo_green ">> Verifying configuration file exists..."
if [ -f "$CONFIG_PATH" ]; then
    echo_green ">> Configuration file found!"
else
    echo ">> ERROR: Configuration file not found at $CONFIG_PATH"
    exit 1
fi

if [ -n "$ORG_ID" ]; then
  python -m hivemind_exp.gsm8k.train_single_gpu \
    --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
    --identity_path "$IDENTITY_PATH" \
    --modal_org_id "$ORG_ID" \
    --contract_address "$SWARM_CONTRACT" \
    --config "$CONFIG_PATH" \
    --game "$GAME"
else
  python -m hivemind_exp.gsm8k.train_single_gpu \
    --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
    --identity_path "$IDENTITY_PATH" \
    --public_maddr "$PUB_MULTI_ADDRS" \
    --initial_peers "$PEER_MULTI_ADDRS" \
    --host_maddr "$HOST_MULTI_ADDRS" \
    --config "$CONFIG_PATH" \
    --game "$GAME"
fi

wait  # Keep script running until Ctrl+C 
