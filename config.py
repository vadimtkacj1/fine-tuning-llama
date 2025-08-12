from pathlib import Path

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
MAX_LENGTH = 256
STORED_DIR = Path('./stored')
OUTPUT_DIR = Path('./output_lora')
VALID_SPEAKERS = ('User1', 'User2')

LORA_CONFIG = {
    'r': 8,
    'lora_alpha': 16,
    'target_modules': ["q_proj", "v_proj"],
    'lora_dropout': 0.1
}

TRAINING_CONFIG = {
    'per_device_train_batch_size': 1,
    'num_train_epochs': 1,
    'logging_steps': 1,
    'save_strategy': 'no',
    'fp16': True,
    'remove_unused_columns': False,
    'report_to': [],
}