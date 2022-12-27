import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl

# Import the YOLOv5 model architecture
# from yolov5 import YOLOv5

# Import the requests library for making HTTP requests
import requests


def load_custom_model(url):
    # Make an HTTP GET request to the URL and save the response
    response = requests.get(url)

    # Load the YOLOv5 model with the custom weights
    model = torch.hub.load('ultralytics/yolov5')
    model.load_state_dict(torch.load(response.content))
    model.conf = 0.5  # confidence threshold (0-1)

    # Return the model
    return model


def load_local_model(weights_path):
    # Load the YOLOv5 model with the custom weights
    model = torch.hub.load('./yolov5', 'custom',
                           path=f'./models/{weights_path}', source='local')
    # model.load_state_dict(torch.load(weights_path))
    model.conf = 0.5  # confidence threshold (0-1)

    # Return the model
    return model
