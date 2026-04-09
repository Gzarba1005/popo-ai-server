from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import asyncio
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

# Permitir que Expo (celular) pueda conectarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

model_name = "nateraw/food"

print("Cargando modelo...")

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

model.eval()

print("Modelo cargado correctamente")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

    topk = torch.topk(probs, 5)
    results = []

    for score, idx in zip(topk.values[0], topk.indices[0]):
        label = model.config.id2label[idx.item()]
        results.append({
            "label": label,
            "confidence": float(score.item())
        })

    return {
        "modelId": "food101@local_v1",
        "topK": results
    }