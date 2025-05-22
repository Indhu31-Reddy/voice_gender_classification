# 🎙️ Voice Gender Classifier

A machine learning project that classifies speaker gender (Male/Female) from voice recordings using MFCC features and a Random Forest classifier.

---

## 🚀 Features

- Upload `.wav` file or record live audio
- Extract MFCC-based features using `librosa`
- Predict gender using trained Random Forest model
- Streamlit web app interface

---

## 📁 Folder Structure

voice_classifier/
├── data/ # Raw audio files
│ ├── male/
│ └── female/
├── features/
│ └── voice_features.csv
├── models/
│ └── voice_gender_rf_model.joblib
├── src/
│ ├── extract_features.py
│ └── train_model.py
├── app.py # Predict gender from uploaded audio files
├── app_live.py # Predict gender using live voice recording
├── requirements.txt
└── readme.md


## 📦 Installation

1. Clone this repo:
    ```bash
    git clone https://github.com/your-username/voice_classifier.git
    cd voice_classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have a microphone for `app_live.py`.

## 🚀 Usage

### Train Model

```bash
cd src
python train_model.py
streamlit run app.py
Model
Trained using RandomForestClassifier

Features: 13 MFCCs + mean, std, skew, kurtosis

Notes
Accuracy: ~99%

Model may be biased based on dataset distribution

Improve by collecting more diverse audio data
uture Improvements
Add speaker age detection

Deploy to Streamlit Cloud

Improve handling for female voice misclassification