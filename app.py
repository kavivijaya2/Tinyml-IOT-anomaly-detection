import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# --------------------------------------------------
# TinyML Edge AI Framework for Real-Time IoT Sensor
# Anomaly Detection Without Cloud Dependency
# --------------------------------------------------

st.set_page_config(
    page_title="TinyML Sensor Anomaly Detection",
    page_icon="📡",
    layout="wide"
)

PROJECT_DIR = Path(__file__).parent

@st.cache_resource
def load_model_files():
    model_path = PROJECT_DIR / "tinyml_autoencoder.tflite"
    scaler_path = PROJECT_DIR / "scaler.pkl"
    config_path = PROJECT_DIR / "config.json"

    with open(config_path, "r") as file:
        config = json.load(file)

    scaler = joblib.load(scaler_path)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    return (
        interpreter,
        scaler,
        config,
        model_path.stat().st_size
    )

st.title("📡 TinyML Edge AI Framework")
st.subheader("Real-Time IoT Sensor Anomaly Detection Without Cloud Dependency")
st.caption(
    "Software simulation using a locally loaded TensorFlow Lite model. "
    "No physical IoT hardware is required for this demonstration."
)

try:
    interpreter, scaler, config, model_size = load_model_files()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

threshold = float(config["threshold"])

st.success("TensorFlow Lite model loaded successfully!")

left, right = st.columns(2)
left.metric("Model Size", f"{model_size / 1024:.2f} KB")
right.metric("Anomaly Threshold", f"{threshold:.8f}")

st.divider()
st.header("🧪 Sensor Input Simulator")
st.write("Enter a sensor reading to simulate a new measurement.")

sensor_value = st.number_input(
    "Sensor temperature value",
    min_value=-100.0,
    max_value=200.0,
    value=25.0,
    step=0.1
)

previous_value = st.number_input(
    "Previous sensor value",
    min_value=-100.0,
    max_value=200.0,
    value=24.8,
    step=0.1
)

rolling_mean = st.number_input(
    "24-reading rolling mean",
    min_value=-100.0,
    max_value=200.0,
    value=25.0,
    step=0.1
)

rolling_std = st.number_input(
    "24-reading rolling standard deviation",
    min_value=0.0,
    max_value=100.0,
    value=0.5,
    step=0.1
)

if st.button("🔍 Detect Anomaly", type="primary"):
    value_change = sensor_value - previous_value
    deviation = sensor_value - rolling_mean

    feature_names = [
        "value",
        "value_change",
        "rolling_mean",
        "rolling_std",
        "deviation_from_mean"
    ]

    features = pd.DataFrame(
        [[
            sensor_value,
            value_change,
            rolling_mean,
            rolling_std,
            deviation
        ]],
        columns=feature_names
    )

    try:
        scaled_features = scaler.transform(features).astype(np.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(
            input_details[0]["index"],
            scaled_features
        )

        start_time = time.perf_counter()
        interpreter.invoke()
        reconstructed = interpreter.get_tensor(
            output_details[0]["index"]
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        error_value = float(
            np.mean((scaled_features - reconstructed) ** 2)
        )

        is_anomaly = error_value > threshold

        st.divider()
        st.header("📊 Detection Results")

        result_col, error_col, time_col = st.columns(3)

        result_col.metric(
            "Prediction",
            "⚠️ Anomaly" if is_anomaly else "✅ Normal"
        )
        error_col.metric("Reconstruction Error", f"{error_value:.8f}")
        time_col.metric("Inference Time", f"{elapsed_ms:.3f} ms")

        if is_anomaly:
            st.error(
                "The reconstruction error exceeds the configured threshold. "
                "The model has flagged this reading as an anomaly."
            )
        else:
            st.success(
                "The reconstruction error is below the configured threshold. "
                "The model has classified this reading as normal."
            )

        st.info(
            "Inference time measures the local TensorFlow Lite model invocation "
            "in this environment. It is not a measurement of physical edge-device "
            "performance."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Academic software prototype | Simulated sensor input | "
    "Physical edge-device deployment is future scope"
)
