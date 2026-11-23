# RuralCareAI
Integrated symptom-based disease prediction + chest X-ray analysis (MobileNetV2 + Grad-CAM) + rule-based triage, in a Streamlit UI.

## Setup
pip install -r requirements.txt

## Data
1. Kaggle "Disease Symptom Prediction" -> data/dataset.csv
2. Kaggle "Chest X-Ray Images (Pneumonia)" -> data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}

## Run
python train_symptom_model.py
python train_image_model.py
streamlit run app.py

## Notes
- Keras 3 note: if loading fails, ensure the same TensorFlow version is used for training and inference.
- Extend: Kannada/Hindi UI, offline TFLite export, other modalities (skin, retina), tele-consult hand-off.
