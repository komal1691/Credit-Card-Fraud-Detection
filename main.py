import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
# -------------------------------------------------
# Initialize FastAPI app
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# Safe Static Folder Mount (only if exists)
# -------------------------------------------------
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Load Model Safely
# -------------------------------------------------
MODEL_PATH = os.path.join(os.getcwd(), "fraud_detection_package.pkl")

try:
    package = joblib.load(MODEL_PATH)
    model = package["model"]
    threshold = package["threshold"]
    features = package["features"]
except Exception as e:
    print("Model loading failed:", e)
    model = None
    threshold = 0.5
    features = []

# -------------------------------------------------
# Templates
# -------------------------------------------------
templates = Jinja2Templates(directory="templates")


# -------------------------------------------------
# Health Check Route (Important for Render)
# -------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "running"}


# -------------------------------------------------
# Home Route
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# -------------------------------------------------
# Prediction Route
# -------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Time: float = Form(...),
    V1: float = Form(...), V2: float = Form(...), V3: float = Form(...),
    V4: float = Form(...), V5: float = Form(...), V6: float = Form(...),
    V7: float = Form(...), V8: float = Form(...), V9: float = Form(...),
    V10: float = Form(...), V11: float = Form(...), V12: float = Form(...),
    V13: float = Form(...), V14: float = Form(...), V15: float = Form(...),
    V16: float = Form(...), V17: float = Form(...), V18: float = Form(...),
    V19: float = Form(...), V20: float = Form(...), V21: float = Form(...),
    V22: float = Form(...), V23: float = Form(...), V24: float = Form(...),
    V25: float = Form(...), V26: float = Form(...), V27: float = Form(...),
    V28: float = Form(...),
    Amount: float = Form(...)
):

    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Model not loaded ❌",
                "probability": "N/A"
            }
        )

    # Create ordered feature dictionary
    input_dict = {
        "Time": Time,
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5,
        "V6": V6, "V7": V7, "V8": V8, "V9": V9,
        "V10": V10, "V11": V11, "V12": V12, "V13": V13,
        "V14": V14, "V15": V15, "V16": V16, "V17": V17,
        "V18": V18, "V19": V19, "V20": V20, "V21": V21,
        "V22": V22, "V23": V23, "V24": V24, "V25": V25,
        "V26": V26, "V27": V27, "V28": V28,
        "Amount": Amount
    }

    try:
        ordered_data = [input_dict[feature] for feature in features]
        data_df = pd.DataFrame([ordered_data], columns=features)

        prob = model.predict_proba(data_df)[0][1]
        prediction = 1 if prob >= threshold else 0

        result = "Fraud Transaction 🚨" if prediction == 1 else "Legitimate Transaction ✅"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "probability": round(float(prob), 4)
            }
        )

    except Exception as e:
        print("Prediction error:", e)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Prediction Error ❌",
                "probability": "N/A"
            }
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, debug=True)



    