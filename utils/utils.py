from config import config


def is_disable_credit_mode_enabled(prompt: str):
    return bool(prompt and config.disable_credit_key in prompt)
