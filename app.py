"""RuralCareAI - run with:  streamlit run app.py"""
import os, joblib, numpy as np, streamlit as st
from PIL import Image
from triage import triage
from gradcam import prep, gradcam, overlay

st.set_page_config(page_title="RuralCareAI", page_icon="🩺", layout="wide")
st.title("🩺 RuralCareAI")
st.caption("AI-based healthcare assistance & medical image analysis — decision support only, not a diagnosis.")

@st.cache_resource
def load_symptom():
    p = "models/symptom_model.joblib"
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_xray():
    p = "models/xray_model.keras"
    if not os.path.exists(p): return None
    import tensorflow as tf
    return tf.keras.models.load_model(p)

sym, xray = load_symptom(), load_xray()
if "xray_prob" not in st.session_state: st.session_state.xray_prob = None

tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Chest X-ray Analysis", "Summary & Referral"])

with tab1:
    if sym is None:
        st.warning("Run train_symptom_model.py first.")
    else:
        pretty = {s.replace("_", " "): s for s in sym["symptoms"]}
        chosen = st.multiselect("Select symptoms", sorted(pretty))
        age = st.number_input("Age", 0, 120, 30)
        if st.button("Predict") and chosen:
            keys = [pretty[c] for c in chosen]
            vec = np.zeros((1, len(sym["symptoms"])))
            for k in keys: vec[0, sym["symptoms"].index(k)] = 1
            probs = sym["model"].predict_proba(vec)[0]
            top = np.argsort(probs)[::-1][:3]
            st.subheader("Possible conditions")
            for i in top:
                st.write(f"**{sym['model'].classes_[i]}** — {probs[i]*100:.1f}%")
                st.progress(float(probs[i]))
            st.session_state.update(symptoms=keys, top_prob=float(probs[top[0]]),
                                    top_disease=sym["model"].classes_[top[0]], age=age)

with tab2:
    if xray is None:
        st.warning("Run train_image_model.py first.")
    else:
        up = st.file_uploader("Upload chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up)
            arr, x = prep(img)
            p = float(xray.predict(x, verbose=0)[0, 0])
            st.session_state.xray_prob = p
            c1, c2 = st.columns(2)
            c1.image(arr.astype("uint8"), caption="Input", use_container_width=True)
            c2.image(overlay(arr, gradcam(xray, x)), caption="Grad-CAM (regions influencing prediction)",
                     use_container_width=True)
            st.metric("Pneumonia probability", f"{p*100:.1f}%")

with tab3:
    ss = st.session_state
    if "symptoms" not in ss and ss.xray_prob is None:
        st.info("Complete the symptom checker and/or X-ray analysis first.")
    else:
        level, advice = triage(ss.get("symptoms", []), ss.get("top_prob", 0.0), ss.xray_prob)
        color = {"EMERGENCY": "red", "URGENT": "orange", "ROUTINE": "green"}[level]
        st.markdown(f"### Triage level: :{color}[{level}]")
        st.write(advice)
        if "top_disease" in ss: st.write(f"Most likely (symptoms): **{ss.top_disease}**")
        if ss.xray_prob is not None: st.write(f"X-ray pneumonia probability: **{ss.xray_prob*100:.1f}%**")
        st.caption("⚠️ This tool supports, and never replaces, a qualified medical professional.")
