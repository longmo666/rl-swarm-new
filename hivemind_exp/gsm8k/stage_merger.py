import logging
from typing import Any, Dict


def merge_stage1_question(outputs: Dict[str, Dict[str, Any]], log_tag=None):
    logger = logging.getLogger(f"{__name__}:{log_tag}")

    # TODO: Currently question+answer keeps getting replaced at every file. This is wasteful and can be optimized
    # TODO: If an agent answers more than once (or >1 answer from the same agent id hash), then current implementation will only keep the last seen in the loop. Should allow for multiple answers?
    merged = {"question": None, "answer": None, "agent_answers": {}}
    for agent, o in outputs.items():
        if not all(k in o for k in merged):
            logger.warning(f"Skipped malformed stage1 output from {agent}: {o}")
            continue
        if not isinstance(o["agent_answers"], dict):
            logger.warning(f"Skipped malformed agent_answers from {agent}: {o}")
            continue
        merged["question"] = o["question"]
        merged["answer"] = o["answer"]
        merged["agent_answers"].update(o["agent_answers"])
    # Fill with default values. TODO: Decide if this is a good choice.
    for agent in outputs:
        if agent not in merged["agent_answers"]:
            merged["agent_answers"][agent] = "No answer received..."
    return merged


def merge_stage2_question(outputs: Dict[str, Dict[str, Any]], log_tag=None):
    logger = logging.getLogger(f"{__name__}:{log_tag}")

    # TODO: Currently question+answer keeps getting replaced at every file. This is wasteful and can be optimized
    # TODO: If an agent answers more than once (or >1 answer from the same agent id hash), then current implementation will only keep the last seen in the loop. Should allow for multiple answers?
    merged = {
        "question": None,
        "answer": None,
        "stage2_prompt": None,
        "agent_opinion": {},
    }
    for agent, o in outputs.items():
        expected_keys = {"question", "answer", "stage2_prompt", "agent_opinion"}
        if not expected_keys.issubset(o.keys()):
            logger.warning(f"Skipped malformed stage2 output from {agent}: {o}")
            continue
        if not isinstance(o["agent_opinion"], dict):
            logger.warning(f"Skipped malformed agent_opinion from {agent}: {o}")
            continue
        merged["question"] = o["question"]
        merged["answer"] = o["answer"]
        merged["stage2_prompt"] = o["stage2_prompt"]
        merged["agent_opinion"].update(o["agent_opinion"])
    # Fill with default values. TODO: Decide if this is a good choice.
    for agent in outputs:
        if agent not in merged["agent_opinion"]:
            merged["agent_opinion"][agent] = "No feedback received..."
    return merged
