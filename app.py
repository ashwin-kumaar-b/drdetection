import io
import os
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import numpy as np

import onnxruntime as ort
import cv2
from pathlib import Path

app = FastAPI(title="DR Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def resolve_model_path() -> str | None:
    # 1. Check environment variable
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    
    # 2. Check local directory for standard model names
    here = Path(__file__).resolve().parent
    candidates = [
        here / "dr_model_fixed.onnx",
        here / "dr_model.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
            
    # 3. Check parent/wildcard paths
    patterns = [
        str(here / "*.onnx"),
        str(here.parent / "*.onnx"),
    ]
    import glob
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
            
    return None

MODEL_PATH = resolve_model_path()
SESSION: ort.InferenceSession | None = None
onnx_input_name = None
onnx_output_name = None

if MODEL_PATH:
    try:
        SESSION = ort.InferenceSession(MODEL_PATH)
        onnx_input_name = SESSION.get_inputs()[0].name
        onnx_output_name = SESSION.get_outputs()[0].name
        print(f"Loaded ONNX model: {MODEL_PATH}")
    except Exception as e:
        print(f"ONNX model load error: {e}")
        SESSION = None
else:
    print("ONNX model not found. Please place dr_model_fixed.onnx in the same directory.")


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

SEVERITY_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected. The retinal examination appears normal with no visible microaneurysms, hemorrhages, or other diabetic changes.",
    1: "Mild Non-Proliferative Diabetic Retinopathy (NPDR) detected. Early-stage DR with microaneurysms present. Regular monitoring recommended.",
    2: "Moderate Non-Proliferative Diabetic Retinopathy (NPDR) detected. Multiple microaneurysms, dot and blot hemorrhages, or cotton wool spots may be present. Closer follow-up advised.",
    3: "Severe Non-Proliferative Diabetic Retinopathy (NPDR) detected. Extensive intraretinal hemorrhages, venous beading, or intraretinal microvascular abnormalities present. Requires prompt ophthalmologic evaluation.",
    4: "Proliferative Diabetic Retinopathy (PDR) detected. Advanced stage with neovascularization and high risk of vision loss. Urgent ophthalmologic consultation required.",
}

def preprocess_image(image: Image.Image):
    img = image.resize((512, 512))
    img_array = np.array(img)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_array = img_array.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = img_array.astype(np.float32).transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


class PredictResponse(BaseModel):
    report: str
    label: str
    confidence: float
    scores: Dict[str, float]


@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": SESSION is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if SESSION is None:
        raise HTTPException(status_code=500, detail="ONNX model not loaded on server")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Preprocess and run ONNX inference
    input_data = preprocess_image(image)
    outputs = SESSION.run([onnx_output_name], {onnx_input_name: input_data})[0]
    
    # Apply softmax to raw scores
    exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probabilities = exp_outputs / exp_outputs.sum(axis=1, keepdims=True)
    probabilities = probabilities[0]
    
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class]) * 100
    max_prob = float(np.max(probabilities))
    second_prob = float(np.partition(probabilities, -2)[-2])
    confidence_gap = max_prob - second_prob

    # Build detailed clinical report
    if max_prob < 0.5 or confidence_gap < 0.15:
        result_text = (
            f"⚠️ LOW CONFIDENCE PREDICTION\n\n"
            f"Classification: {CLASS_NAMES[predicted_class]}\n"
            f"Confidence Level: {confidence:.1f}%\n\n"
            f"IMPORTANT NOTICE:\n"
            f"The model's confidence for this prediction is below the recommended threshold. This may indicate:\n"
            f"• Image quality may be insufficient for accurate analysis\n"
            f"• The retinal scan may require professional preprocessing\n\n"
            f"Please ensure you have uploaded a clear, well-lit retinal fundus photograph. "
            f"For accurate diagnosis, consult directly with an ophthalmologist."
        )
    else:
        certainty = "high" if confidence >= 85 else "moderate" if confidence >= 70 else "fair"
        result_text = (
            f"DIAGNOSTIC ANALYSIS REPORT\n\n"
            f"PRIMARY DIAGNOSIS: {CLASS_NAMES[predicted_class]}\n"
            f"Confidence Level: {confidence:.1f}%\n\n"
            f"CLINICAL FINDINGS:\n"
            f"{SEVERITY_DESCRIPTIONS[predicted_class]}\n\n"
            f"ANALYSIS SUMMARY:\n"
            f"The AI model has analyzed the retinal fundus image and identified features consistent with {CLASS_NAMES[predicted_class]} diabetic retinopathy. "
            f"The confidence level of {confidence:.1f}% indicates a {certainty} degree of diagnostic certainty.\n\n"
            f"NEXT STEPS:\n"
            f"• This AI screening should be followed by a comprehensive ophthalmologic examination.\n"
            f"• Regular monitoring and blood glucose management remain essential.\n"
            f"• Consult with your healthcare provider regarding appropriate options."
        )

    return {
        "report": result_text,
        "label": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "scores": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(probabilities))},
    }
