from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import os
import uuid
import json

# Initialize FastAPI app
app = FastAPI()

# Directories
UPLOAD_DIR = "storage/uploads"
PROCESSED_DIR = "storage/processed"
LOG_DIR = "storage/logs"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Utility to write log file
def write_log(file_id, challenge, log_data):
    log_file_path = os.path.join(LOG_DIR, f"{file_id}_{challenge}.json")
    with open(log_file_path, "w") as log_file:
        json.dump(log_data, log_file, indent=4)
    return log_file_path

@app.post("/process")
async def process_image(
    challenge: str = Form(...),
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    shadows: bool = Form(False),
    background_type: str = Form(None),
    banner_text: str = Form(None),
    font_size: int = Form(None),
    text_position: int = Form(None),
    lifestyle_type: str = Form(None)
):
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        original_file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(original_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Load image for processing
        image = Image.open(original_file_path)

        # Pass to challenge-specific function
        if challenge == "Challenge 1: Foundation Enhancement":
            from challenge_1 import foundation_enhancement
            processed_image, log_data = foundation_enhancement(
                image, brightness, contrast, shadows
            )
        elif challenge == "Challenge 2: Background Integration":
            from challenge_2 import background_integration
            processed_image, log_data = background_integration(image, background_type)
        elif challenge == "Challenge 3: Text and Banner Integration":
            from challenge_3 import text_banner_integration
            processed_image, log_data = text_banner_integration(
                image, banner_text, font_size, text_position
            )
        elif challenge == "Challenge 4: Lifestyle Context Creation":
            from challenge_4 import lifestyle_context
            processed_image, log_data = lifestyle_context(image, lifestyle_type)
        elif challenge == "Challenge 5: Advanced Composition":
            from challenge_5 import advanced_composition
            processed_image, log_data = advanced_composition(image)
        else:
            return JSONResponse(
                status_code=400, content={"error": "Invalid challenge selected"}
            )

        # Save processed image
        processed_file_path = os.path.join(PROCESSED_DIR, f"{file_id}_{challenge}.jpg")
        processed_image.save(processed_file_path)

        # Write log file
        log_file_path = write_log(file_id, challenge, log_data)

        # Return paths to frontend
        return {
            "image_url": processed_file_path,
            "log_url": log_file_path
        }

    except Exception as e:
        error_log = {
            "error": str(e),
            "challenge": challenge,
            "file": file.filename
        }
        log_file_path = write_log(str(uuid.uuid4()), challenge, error_log)
        return JSONResponse(
            status_code=500, content={"error": str(e), "log_url": log_file_path}
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Enhancement Backend"}
