# TinyML Edge AI Framework for Real-Time IoT Sensor Anomaly Detection Without Cloud Dependency

## Project Overview
This project demonstrates a software-based TinyML framework for detecting anomalies in IoT sensor data locally, without depending on cloud services for model inference.

The system uses a trained autoencoder converted to TensorFlow Lite to identify unusual sensor readings based on reconstruction error.

## Objectives
- Simulate IoT sensor data.
- Preprocess sensor readings and extract features.
- Train a lightweight anomaly detection model.
- Convert the trained model to TensorFlow Lite.
- Perform local inference without cloud dependency.
- Display anomaly detection results through an interactive dashboard.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow
- TensorFlow Lite
- Streamlit
- Google Colab

## System Workflow
1. Sensor Dataset
2. Data Preprocessing
3. Feature Extraction
4. TinyML Autoencoder
5. TensorFlow Lite Inference
6. Anomaly Detection
7. Dashboard and Results

## Project Files
- `app.py` — Interactive dashboard
- `requirements.txt` — Required Python packages
- `tinyml_autoencoder.tflite` — Trained TensorFlow Lite model
- `scaler.pkl` — Feature scaling object
- `config.json` — Model configuration and anomaly threshold
- `processed_sensor_data.csv` — Processed sensor dataset

## Dataset
The project uses the ambient temperature system failure dataset from the Numenta Anomaly Benchmark (NAB).

## How to Run Locally
Install the required dependencies:

```bash
pip install -r requirements.txt
