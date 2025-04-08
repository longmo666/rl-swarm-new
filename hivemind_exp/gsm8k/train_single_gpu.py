import logging
import colorlog
import requests  # Make sure requests is imported for Telegram logging
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


def main():
    # Setup logging.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(light_red)s%(levelname)s:%(name)s:%(message)s")
    )
    root_logger.addHandler(handler)

    # Send initial log to Telegram
    send_telegram("Training script started. Initializing configurations...")

    try:
        # Parsing arguments and configs
        parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
        model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()

        # Send configuration info to Telegram
        send_telegram(f"Configurations loaded: {model_args}, {grpo_args}, {testnet_args}, {training_args}")

        # Run main training loop.
        if org_id := testnet_args.modal_org_id:
            send_telegram(f"Starting with ModalSwarmCoordinator for org_id: {org_id}")
            runner = TestnetGRPORunner(ModalSwarmCoordinator(org_id, web3=setup_web3()))
        elif priv_key := testnet_args.wallet_private_key:
            send_telegram(f"Starting with WalletSwarmCoordinator for private_key: {priv_key[:8]}***")
            runner = TestnetGRPORunner(WalletSwarmCoordinator(priv_key, web3=setup_web3()))
        else:
            send_telegram("Starting with GRPORunner")
            runner = GRPORunner()

        # Send a message to Telegram about starting the run
        send_telegram("Running the training...")

        runner.run(model_args, grpo_args, training_args, get_stage1_samples)
        send_telegram("Training completed successfully!")

    except Exception as e:
        send_telegram(f"⚠️ Error occurred during execution: {str(e)}")
        logging.error(f"⚠️ Error: {e}")
    

if __name__ == "__main__":
    main()
