import torch
import numpy as np
from feature_extractor import extract_features

# 15 disorder labels
labels = [
"depression","anxiety","bipolar","ptsd","ocd",
"schizophrenia","adhd","eating_disorder","bpd",
"panic_disorder","social_anxiety","insomnia",
"autism","phobia","stress_disorder"
]

# Model architecture
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(19,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,15),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)

# Load trained model
model = Model()
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
model.eval()


def predict_audio(audio_path):

    # Extract audio features
    features = extract_features(audio_path)

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Model prediction
    pred = model(x).detach().numpy()[0]

    # Create probability dictionary
    probabilities = {}
    for i,label in enumerate(labels):
        probabilities[label] = float(round(pred[i],3))

    # Threshold for predicted disorders
    threshold = 0.5
    predicted_disorders = []

    for i,label in enumerate(labels):
        if pred[i] > threshold:
            predicted_disorders.append(label)

    # Main prediction (highest probability)
    max_index = np.argmax(pred)
    main_prediction = labels[max_index]

    return {
        "probabilities": probabilities,
        "predicted_disorders": predicted_disorders,
        "main_prediction": main_prediction
    }