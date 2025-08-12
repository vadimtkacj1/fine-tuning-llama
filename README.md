# LoRA Fine-Tuning Microservice

## 📌 Overview

This microservice provides:

1. **Upload dialogues** in JSON format and store only the messages from a selected speaker.
2. **Fine-tune a LLaMA model** (such as `meta-llama/Meta-Llama-3-8B-Instruct`) using **LoRA** on saved user messages.

Supported speakers:

- `User1`
- `User2`

---

## 🛠 Technologies

- **Python 3.10+**
- **FastAPI** — REST API framework
- **Hugging Face Transformers** — model loading and training
- **PEFT (LoRA)** — parameter-efficient fine-tuning
- **Datasets** — dataset creation
- **Uvicorn** — ASGI server

---

## 📂 Project Structure

```
.
├── main.py        # FastAPI endpoints
├── train.py       # LoRA training logic
├── parser.py      # Dialogue parsing and saving
├── config.py      # Project configuration
├── stored/        # Stored user messages
├── output_lora/   # Fine-tuned LoRA models
└── requirements.txt
```

---

## ⚙ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vadimtkacj1/fine-tuning-llama
cd project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 --index-url https://download.pytorch.org/whl/cu118 # for GPU
pip install -r requirements.txt
```

---

## 🔑 Using `meta-llama/Meta-Llama-3-8B-Instruct`

The **Meta LLaMA 3** models require authentication with Hugging Face.

1. **Create a Hugging Face account**  
   [Sign up here](https://huggingface.co/join)

2. **Accept the model’s license**  
   Visit: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  
   Click **"Agree to license"**.

3. **Get your access token**

   - Go to: [Hugging Face Tokens](https://huggingface.co/settings/tokens)
   - Create a **Read** token and copy it.

4. **Login to Hugging Face**

```bash
huggingface-cli login
```

Paste your token when prompted.

5. **Update `config.py`**

```python
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
```

---

## 🚀 Running the Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation is available at:

```
http://localhost:8000/docs
```

---

## ⚡ Quick Start

### Development mode (small model for testing)

Edit `config.py`:

```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Run:

```bash
uvicorn main:app --reload
```

This runs quickly and allows testing the pipeline without heavy downloads.

### Production mode (Meta-LLaMA-3)

1. Follow the **Hugging Face token setup** steps above.
2. Edit `config.py`:

```python
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
```

3. Run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📡 REST API Endpoints

### **1️⃣ POST /upload-text**

Uploads a JSON file containing a dialogue and saves only the messages from the specified speaker.

**Form-data parameters:**

- `file` — `.json` file, example:

```json
{
  "dialog": [
    {"speaker": "User1", "content": "Hello! How are you?"},
    {"speaker": "User2", "content": "I’m good, thanks."},
    {"speaker": "User1", "content": "Shall we grab a coffee?"}
  ]
}
```

- `speaker` — `"User1"` or `"User2"`

**Example request:**

```bash
curl -X POST "http://localhost:8000/upload-text"   -F "file=@dialog.json"   -F "speaker=User1"
```

**Example response:**

```json
{
  "status": "ok",
  "saved_path": "stored/User1.json"
}
```

---

### **2️⃣ POST /train**

Starts LoRA fine-tuning for the selected speaker.

**Request body:**

```json
{
  "speaker": "User1"
}
```

**Example request:**

```bash
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d "{\"speaker\": \"User1\"}"
```

**Example response:**

```json
{
  "status": "ok",
  "output_dir": "output_lora/User1"
}
```

---

## ⚙ Configuration

Configuration is stored in `config.py`.

| Parameter         | Description                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `MODEL_NAME`      | Model name (`meta-llama/Meta-Llama-3-8B-Instruct` for production, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for testing) |
| `MAX_LENGTH`      | Maximum token length                                                                                                |
| `VALID_SPEAKERS`  | Allowed speaker names                                                                                               |
| `LORA_CONFIG`     | LoRA settings (`r=8`, `alpha=16`, `target_modules=["q_proj", "v_proj"]`, `dropout=0.1`)                             |
| `TRAINING_CONFIG` | Training parameters (1 epoch, batch size = 1)                                                                       |

---

## 📦 Example `requirements.txt`

```txt
fastapi==0.116.1
uvicorn[standard]==0.35.0
python-multipart==0.0.20
transformers>=4.30.0
peft==0.17.0
accelerate==1.10.0
datasets==4.0.0
bitsandbytes==0.47.0
safetensors==0.6.2
numpy==1.26.4
sentencepiece==0.1.99
protobuf==3.20.3

```

---

## 📜 License

MIT License
