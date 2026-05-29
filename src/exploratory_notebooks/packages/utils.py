# inference_service/utils.py

import pickle
import numpy as np
import librosa

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "BirdNET-Analyzer"))
from birdnet_analyzer.model import load_model, embeddings
load_model(class_output=False)

with open("71_ensemble_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

scaler = metadata["scaler"]
species_to_index = metadata["species_to_index"]
index_to_species = {idx: species for species, idx in species_to_index.items()}
model_paths = metadata["model_paths"]
num_classes = metadata["num_classes"]

models = []
for path in model_paths:
    with open(path, "rb") as f:
        models.append(pickle.load(f))
        

def extract_embeddings(audio_bytes: bytes, sample_rate: int = 48000):
    audio, _ = librosa.load(audio_bytes, sr=sample_rate, mono=True)

    if len(audio) < sample_rate * 3:
        pad = sample_rate * 3 - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    else:
        audio = audio[:sample_rate * 3]

    audio_input = np.expand_dims(audio, axis=0)
    return embeddings(audio_input)

def predict_top_5_frog_species(embedding: np.ndarray):
    scaled = scaler.transform(embedding)
    ensemble_preds = np.zeros((scaled.shape[0], num_classes))

    for model in models:
        ensemble_preds += model.predict_proba(scaled)

    ensemble_preds /= len(models)
    prediction = ensemble_preds[0]
    top_indices = np.argsort(prediction)[-5:][::-1]
    top_species = [index_to_species[i] for i in top_indices]
    top_confidences = prediction[top_indices]

    return list(zip(top_species, top_confidences))
