from transformers import BitsAndBytesConfig
from typing import List
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from config import MODEL_NAME, MAX_LENGTH, OUTPUT_DIR, LORA_CONFIG, TRAINING_CONFIG
from parser import load_stored_messages, build_prompt

def setup_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

def prepare_dataset(messages: List[str], tokenizer: AutoTokenizer) -> Dataset:
    records = [{"text": build_prompt(text)} for text in messages]
    dataset = Dataset.from_list(records)
    
    def tokenize_fn(batch):
        tokenized = tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    return dataset

def setup_model() -> PeftModel:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    load_kwargs = {
        'device_map': 'auto',
        'quantization_config': quant_config,
    }

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    return model

def run_training(speaker: str) -> str:
    # Load and prepare data
    messages = load_stored_messages(speaker)
    tokenizer = setup_tokenizer()
    dataset = prepare_dataset(messages, tokenizer)
    
    # Setup model
    model = setup_model()
    
    # Prepare output directory
    output_dir = OUTPUT_DIR / speaker
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **TRAINING_CONFIG
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )
    
    # Run training
    trainer.train()
    model.save_pretrained(str(output_dir))
    
    return str(output_dir)