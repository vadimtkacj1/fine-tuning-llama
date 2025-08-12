import json
from pathlib import Path
from typing import List
from config import STORED_DIR, VALID_SPEAKERS

def validate_speaker(speaker: str) -> None:
    if speaker not in VALID_SPEAKERS:
        raise ValueError(f'speaker must be one of {VALID_SPEAKERS}')

def build_prompt(text: str) -> str:
    return text

def load_stored_messages(speaker: str) -> List[str]:
    validate_speaker(speaker)
    
    stored_path = STORED_DIR / f'{speaker}.json'
    if not stored_path.exists():
        raise FileNotFoundError(
            f'Stored messages for {speaker} not found. Upload them first via /upload-text'
        )
    
    with stored_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    
    messages: List[str] = payload.get('messages', [])
    if not messages:
        raise ValueError('No messages found to train on')
    
    return messages

def save_speaker_messages(raw_bytes: bytes, speaker: str) -> Path:
    try:
        payload = json.loads(raw_bytes)
    except Exception:
        raise ValueError('Invalid JSON file')

    if 'dialog' not in payload or not isinstance(payload['dialog'], list):
        raise ValueError('JSON must contain a "dialog" array')

    validate_speaker(speaker)

    messages = [
        item.get('content', '') 
        for item in payload['dialog'] 
        if item.get('speaker') == speaker
    ]

    STORED_DIR.mkdir(exist_ok=True)
    out_path = STORED_DIR / f'{speaker}.json'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"speaker": speaker, "messages": messages}, 
            f, 
            ensure_ascii=False, 
            indent=2
        )

    return out_path