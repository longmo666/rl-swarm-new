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
            logging.info(f"📬 Telegram message sent: {msg}")
        except Exception as e:
            logging.warning(f"📬 Telegram failed: {e}")


def auto_adjust_batch_size(training_args):
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        approx_batch_size = max(1, min(4, int(free_mem / 1e9)))
        training_args.per_device_train_batch_size = approx_batch_size
        logging.info(f"🧠 Auto batch size adjusted to: {approx_batch_size}")
        send_telegram(f"🧠 Auto batch size adjusted to: {approx_batch_size}")
    except Exception as e:
        logging.warning(f"Batch auto-adjust failed: {e}")
        send_telegram(f"⚠️ Batch auto-adjust failed: {e}")


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

    logging.info("🚀 Starting auto batch size adjustment...")
    send_telegram("🚀 Starting auto batch size adjustment...")
    auto_adjust_batch_size(training_args)

    if org_id := testnet_args.modal_org_id:
        runner = TestnetGRPORunner(ModalSwarmCoordinator(org_id, web3=setup_web3()))
        logging.info(f"📱 Running with ModalSwarmCoordinator. Org ID: {org_id}")
        send_telegram(f"📱 Running with ModalSwarmCoordinator. Org ID: {org_id}")
    elif priv_key := testnet_args.wallet_private_key:
        runner = TestnetGRPORunner(WalletSwarmCoordinator(priv_key, web3=setup_web3()))
        logging.info(f"📱 Running with WalletSwarmCoordinator. Private Key: {priv_key[:5]}... (for security)")
        send_telegram(f"📱 Running with WalletSwarmCoordinator. Private Key: {priv_key[:5]}... (for security)")
    else:
        runner = GRPORunner()
        logging.info("📱 Running with GRPORunner.")
        send_telegram("📱 Running with GRPORunner.")

    send_telegram("🚀 Starting *Qwen 2.5* fine-tune...")

    try:
        logging.info("🔧 Starting training...")
        send_telegram("🔧 Starting training...")
        runner.run(model_args, grpo_args, training_args, get_stage1_samples)
        logging.info("✅ Fine-tuning completed successfully.")
        send_telegram("✅ *Qwen 2.5* fine-tune completed successfully.")
    except torch.cuda.OutOfMemoryError:
        logging.warning("⚠️ CUDA OOM! Retrying...")
        send_telegram("⚠️ CUDA OOM! Retrying with smaller batch size...")
        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps *= 2
        logging.info(f"🧠 Batch size reduced to {training_args.per_device_train_batch_size}, "
                     f"gradient accumulation steps increased to {training_args.gradient_accumulation_steps}.")
        send_telegram(f"🧠 Batch size reduced to {training_args.per_device_train_batch_size}, "
                       f"gradient accumulation steps increased to {training_args.gradient_accumulation_steps}.")
        torch.cuda.empty_cache()

        try:
            logging.info("🔧 Retry training...")
            send_telegram("🔧 Retry training...")
            runner.run(model_args, grpo_args, training_args, get_stage1_samples)
            logging.info("✅ Retry successful. Fine-tune completed.")
            send_telegram("✅ Retry successful. Fine-tune completed.")
        except Exception as e:
            logging.error(f"❌ Retry failed: {e}")
            send_telegram(f"❌ Retry failed: {e}")

    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        send_telegram(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()
