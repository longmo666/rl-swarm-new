# import logging

# import colorlog
# from trl import GRPOConfig, ModelConfig, TrlParser

# from hivemind_exp.chain_utils import (
#     ModalSwarmCoordinator,
#     WalletSwarmCoordinator,
#     setup_web3,
# )
# from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples
# from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
#     TestnetGRPOArguments,
#     TestnetGRPORunner,
# )
# from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner


# def main():
#     # Setup logging.
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.INFO)
#     handler = colorlog.StreamHandler()
#     handler.setFormatter(
#         colorlog.ColoredFormatter("%(light_red)s%(levelname)s:%(name)s:%(message)s")
#     )
#     root_logger.addHandler(handler)

#     parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
#     model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()

#     # Run main training loop.
#     if org_id := testnet_args.modal_org_id:
#         runner = TestnetGRPORunner(ModalSwarmCoordinator(org_id, web3=setup_web3()))
#     elif priv_key := testnet_args.wallet_private_key:
#         runner = TestnetGRPORunner(WalletSwarmCoordinator(priv_key, web3=setup_web3()))
#     else:
#         runner = GRPORunner()

#     runner.run(model_args, grpo_args, training_args, get_stage1_samples)


# if __name__ == "__main__":
#     main()


import logging
import os
import requests

import colorlog
import torch
from trl import GRPOConfig, ModelConfig, TrlParser

from hivemind_exp.chain_utils import (
    ModalSwarmCoordinator,
    WalletSwarmCoordinator,
    setup_web3,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples
from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
    TestnetGRPOArguments,
    TestnetGRPORunner,
)
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner


def send_telegram(msg: str):
    token = "7700324344:AAFHqwf7jmf00hGIRDjOBTv83evNF1izYVk"
    chat_id = "-1002234568251"
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
        except Exception as e:
            logging.warning(f"📬 Telegram failed: {e}")


def auto_adjust_batch_size(training_args):
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        approx_batch_size = max(1, min(4, int(free_mem / 1e9)))
        training_args.per_device_train_batch_size = approx_batch_size
        logging.info(f"🧠 Auto batch size adjusted to: {approx_batch_size}")
    except Exception as e:
        logging.warning(f"Batch auto-adjust failed: {e}")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(light_red)s%(levelname)s:%(name)s:%(message)s")
    )
    root_logger.addHandler(handler)

    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()

    # Optional 8-bit memory optimization 
    from transformers import BitsAndBytesConfig
    model_args.quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    auto_adjust_batch_size(training_args)

    if org_id := testnet_args.modal_org_id:
        runner = TestnetGRPORunner(ModalSwarmCoordinator(org_id, web3=setup_web3()))
    elif priv_key := testnet_args.wallet_private_key:
        runner = TestnetGRPORunner(WalletSwarmCoordinator(priv_key, web3=setup_web3()))
    else:
        runner = GRPORunner()

    send_telegram("\ud83d\ude80 Starting *Qwen 2.5* fine-tune...")

    try:
        runner.run(model_args, grpo_args, training_args, get_stage1_samples)
        send_telegram("\u2705 *Qwen 2.5* fine-tune completed successfully.")
    except torch.cuda.OutOfMemoryError:
        logging.warning("\u26a0\ufe0f CUDA OOM! Retrying...")
        send_telegram("\u26a0\ufe0f CUDA OOM! Retrying with smaller batch size...")
        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps *= 2
        torch.cuda.empty_cache()
        runner.run(model_args, grpo_args, training_args, get_stage1_samples)
        send_telegram("\u2705 Retry successful. Fine-tune completed.")


if __name__ == "__main__":
    main()
