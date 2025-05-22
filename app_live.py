import streamlit as st
import sounddevice as sd
import wavio
import numpy as np
import joblib
import os
import sys

# Add src folder to path to import extract_features
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from extract_features import extract_features

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models", "voice_gender_rf_model.joblib")
model = joblib.load(model_path)

# Label map, ensure this matches training labels exactly!
label_map = {1: "male", 0: "female"}

def record_audio(duration=3, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("Recording complete!")
    return np.squeeze(recording), fs

def save_audio(data, fs, filename="temp_audio.wav"):
    wavio.write(filename, data, fs, sampwidth=2)
    return filename

def main():
    st.title("Live Voice Gender Classification")

    duration = st.slider("Recording Duration (seconds)", 1, 10, 3)

    if st.button("Record"):
        audio_data, fs = record_audio(duration=duration)
        audio_file = save_audio(audio_data, fs)

        # Extract features
        features = extract_features(audio_file)

        st.write(f"Extracted features shape: {features.shape}")
        st.write(f"Sample features (first 5 values): {features[:5]}")

        # Predict probabilities to understand model confidence
        probabilities = model.predict_proba([features])[0]
        st.write(f"Prediction probabilities: {probabilities}")

        prediction = model.predict([features])[0]
        st.write(f"Raw prediction label: {prediction}")

        predicted_gender = label_map.get(prediction, "Unknown")
        st.success(f"Predicted Gender: {predicted_gender}")

        try:
            os.remove(audio_file)
        except Exception as e:
            st.warning(f"Failed to delete temp audio file: {e}")

if __name__ == "__main__":
    main()
