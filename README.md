# CarboBuddy AI Server — API de reconocimiento de comidas

Servidor de inferencia en **Python + FastAPI** que recibe la foto de un plato y devuelve los 5 alimentos más probables con su nivel de confianza. Fue el primer motor de detección de [CarboBuddy](https://github.com/Gzarba1005/popo-mvp), una app de cálculo de insulina para personas con diabetes tipo 1.

## Cómo funciona

- Modelo **Vision Transformer (ViT)** preentrenado en el dataset **Food-101** ([`nateraw/food`](https://huggingface.co/nateraw/food)), cargado con Hugging Face Transformers y PyTorch.
- La imagen se preprocesa, pasa por el modelo y se aplica *softmax* para obtener probabilidades.
- Devuelve el **top-5** de clases con su confianza.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `POST` | `/predict` | Recibe una imagen (`multipart/form-data`, campo `file`) y devuelve las predicciones |

Ejemplo de respuesta:

```json
{
  "modelId": "food101@local_v1",
  "topK": [
    { "label": "pizza", "confidence": 0.93 },
    { "label": "lasagna", "confidence": 0.03 }
  ]
}
```

## Correr localmente

```bash
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Probar:

```bash
curl -F "file=@plato.jpg" http://localhost:8000/predict
```

## Stack

Python · FastAPI · Uvicorn · PyTorch · Hugging Face Transformers · Pillow

## Evolución

Este servidor se reemplazó luego por un proxy en Cloudflare Workers con la API de visión de Claude: menor costo de mantenimiento, sin servidor propio y mayor velocidad.
