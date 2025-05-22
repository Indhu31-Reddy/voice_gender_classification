import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)
    mfccs_var = mfccs.var(axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_var = chroma.var(axis=1)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = mel.mean(axis=1)
    mel_var = mel.var(axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = contrast.mean(axis=1)
    contrast_var = contrast.var(axis=1)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1)
    tonnetz_var = tonnetz.var(axis=1)

    # Combine all features
    features = np.hstack([
        mfccs_mean, mfccs_var,
        chroma_mean, chroma_var,
        mel_mean, mel_var,
        contrast_mean, contrast_var,
        tonnetz_mean, tonnetz_var
    ])

    return features

def main():
    st.title("Voice Gender Classification")

    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
    if uploaded_file is not None:
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_file)

        features = extract_features(temp_file)

        # Load feature columns from the CSV to maintain correct order
        script_dir = os.path.dirname(os.path.abspath(__file__))
        features_csv_path = os.path.join(script_dir, 'features', 'voice_features.csv')
        feature_df = pd.read_csv(features_csv_path)
        feature_columns = feature_df.columns.drop('label')

        # Create DataFrame with features and correct columns
        features_df = pd.DataFrame([features], columns=feature_columns)

        # Load the trained model
        model_path = os.path.join(script_dir, 'models', 'voice_gender_rf_model.joblib')
        model = joblib.load(model_path)

        # Predict gender
        prediction = model.predict(features_df)[0]
        gender = "Male" if prediction == 1 else "Female"

        st.write(f"**Predicted Gender:** {gender}")

if __name__ == "__main__":
    main()
