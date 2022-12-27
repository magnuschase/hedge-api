from fastapi import FastAPI
from functions.load_model import load_local_model, load_custom_model
from starlette.responses import Response
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from pydantic import BaseModel

app = FastAPI(
    title="YOLOv5 Object Detection API",
    description="Obtain object detection predictions from a YOLOv5 model",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get('/health')
def health():
    return dict(msg='WORKING')


class PredictBody(BaseModel):
    imageUrl: str  # image download URL
    modelPath: str  # path to model
    modelType: str  # local or custom


@app.post('/predict-json')
async def predict_json(body: PredictBody):
    # Load the model
    if body.modelType == 'local':
        model = load_local_model(body.modelPath)
    elif body.modelType == 'custom':
        model = load_custom_model(body.modelPath)
    else:
        return Response(status_code=400, content="Invalid model type")

    # Make an HTTP GET request to the URL and save the response
    response = requests.get(body.imageUrl)

    # Load the image
    image = Image.open(io.BytesIO(response.content))

    # Make a prediction
    results = model(image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"results": detect_res}


@app.post("/predict")
async def predict(body: PredictBody):
    # Load the model
    if body.modelType == 'local':
        model = load_local_model(body.modelPath)
    elif body.modelType == 'custom':
        model = load_custom_model(body.modelPath)
    else:
        return Response(status_code=400, content="Invalid model type")

    # Make an HTTP GET request to the URL and save the response
    response = requests.get(body.imageUrl)

    # Load the image
    image = Image.open(io.BytesIO(response.content))

    # Make a prediction
    results = model(image)
    results.render()
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format='jpeg')

    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
