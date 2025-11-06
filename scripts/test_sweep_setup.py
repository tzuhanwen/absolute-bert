import logging

from transformers import AutoTokenizer

from absolute_bert.utils import init_logging

logger = logging.getLogger(__name__)

def main():
    init_logging()

    from absolute_bert.sweep import setup

    config_unresolved = setup.get_config()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    config = config_unresolved.resolve(tokenizer.vocab_size)
    logger.info(f"config resolved: {config=}")