import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.main_pipeline import run_pipeline
from src.utils.database import log_analysis
from src.utils.config import INPUT_DIR

app = FastAPI(title="Handwriting Personality AI API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(INPUT_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Handwriting Personality AI API is running."}

@app.post("/predict")
async def predict_personality(
    file: UploadFile = File(...),
    use_deep_features: bool = Form(True),
    save_outputs: bool = Form(True),
    is_signature: bool = Form(False)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Save uploaded file
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run pipeline
        result = run_pipeline(
            image_path=file_path,
            use_deep_features=use_deep_features,
            save_outputs=save_outputs,
            is_signature=is_signature
        )
        
        # Note: The database logging is already handled inside run_pipeline!
        
        return {
            "status": "success",
            "filename": file.filename,
            "personality": {
                "scores": result["personality"]["scores"],
                "labels": result["personality"]["labels"],
                "method": result["personality"]["method"]
            },
            "output_paths": result["output_paths"],
            "elapsed_sec": result["elapsed_sec"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
