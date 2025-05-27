# Sentiment Analysis Project

This project is an emotion detection model built using a DistilBERT-based architecture. It uses the GoEmotions dataset for training and provides fast, lightweight sentiment analysis.

---

## Project Structure

- `best_model.pt` - Trained model weights (tracked with Git LFS)
- `goemotions_model/model.safetensors` - Additional model files (tracked with Git LFS)
- `sentiment_analysis.py` - Main script for running inference
- `requirements.txt` - Python dependencies
- `README.md` - This file

---

## Features

- Emotion detection using a pre-trained DistilBERT model
- Lightweight and quantized for fast loading and inference
- Supports multiple emotions from the GoEmotions dataset

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Adnan00707/emotion-detection-app.git
   cd emotion-detection-app
