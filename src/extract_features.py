import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        # Use statistical summaries (mean and std)
        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(mel, axis=1), np.std(mel, axis=1),
            np.mean(contrast, axis=1), np.std(contrast, axis=1),
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
        ])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'features')
    os.makedirs(output_dir, exist_ok=True)

    features = []
    labels = []

    for gender in ['male', 'female']:
        folder = os.path.join(data_dir, gender)
        label = 1 if gender == 'male' else 0

        print(f"Processing {gender} folder...")

        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                filepath = os.path.join(folder, filename)
                feat = extract_features(filepath)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
                    print(f"Processed: {filename}")

    df = pd.DataFrame(features)
    df['label'] = labels

    output_path = os.path.join(output_dir, 'voice_features.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Features saved to: {output_path}")

if __name__ == "__main__":
    main()
