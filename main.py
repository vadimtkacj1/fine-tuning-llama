from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from parser import save_speaker_messages
from train import run_training
from pydantic import BaseModel
import uvicorn

class TrainRequest(BaseModel):
    speaker: str

app = FastAPI(docs_url="/docs")

@app.post('/upload-text')
async def upload_text(file: UploadFile = File(...), speaker: str = Form(...)):
    body = await file.read()
    
    try:
        saved_path = save_speaker_messages(body, speaker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse({
        "status": "ok", 
        "saved_path": str(saved_path)
    })

@app.post('/train')
async def train_endpoint(data: TrainRequest):
    try:
        output_path = run_training(data.speaker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    return JSONResponse({
        "status": "ok", 
        "output_dir": output_path
    })

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)