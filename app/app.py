import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# PAGE CONFIGURATION — keep the existing UI
st.set_page_config(
    page_title="AI Image Forensics Detector",
    page_icon="🔍",
    layout="centered"
)

# MODEL PATHS
CIFAKE_MODEL_PATH = "/content/cifake_efficientnetb0_best.keras"
CASIA_MODEL_PATH = "/content/casia_improved_stage1_best.keras"

# LOAD MODELS
@st.cache_resource
def load_models():
    if not os.path.exists(CIFAKE_MODEL_PATH):
        raise FileNotFoundError(
            f"CIFAKE model not found: {CIFAKE_MODEL_PATH}"
        )

    if not os.path.exists(CASIA_MODEL_PATH):
        raise FileNotFoundError(
            f"CASIA model not found: {CASIA_MODEL_PATH}"
        )

    cifake_model = tf.keras.models.load_model(
        CIFAKE_MODEL_PATH, compile=False
    )

    casia_model = tf.keras.models.load_model(
        CASIA_MODEL_PATH, compile=False
    )

    return cifake_model, casia_model


cifake_model, casia_model = load_models()

# CONSTANTS
IMG_SIZE = 224
AI_THRESHOLD = 0.50
TAMPER_THRESHOLD = 0.50

# IMAGE PREPROCESSING
def preprocess_image(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image).astype(np.float32)
    return np.expand_dims(image_array, axis=0)


# CIFAKE PREDICTION
# FAKE = 0, REAL = 1
# Sigmoid output represents REAL probability.
def predict_cifake(image):
    processed = preprocess_image(image)

    prediction = float(
        cifake_model.predict(processed, verbose=0)[0][0]
    )

    real_probability = float(np.clip(prediction, 0.0, 1.0))
    fake_probability = 1.0 - real_probability

    return fake_probability, real_probability


# CASIA PREDICTION
# AUTHENTIC = 0, TAMPERED = 1
# Sigmoid output represents TAMPERED probability.
def predict_casia(image):
    processed = preprocess_image(image)

    prediction = float(
        casia_model.predict(processed, verbose=0)[0][0]
    )

    tampered_probability = float(np.clip(prediction, 0.0, 1.0))
    authentic_probability = 1.0 - tampered_probability

    return tampered_probability, authentic_probability


# BOTH MODELS ALWAYS RUN
def analyze_image(image):

    fake_prob, real_prob = predict_cifake(image)

    tampered_prob, authentic_prob = predict_casia(image)

    # FINAL DECISION
    if fake_prob >= AI_THRESHOLD:

        final_result = "AI-GENERATED IMAGE"
        confidence = fake_prob
        stage = "CIFAKE AI-GENERATED DETECTOR"

    elif tampered_prob >= TAMPER_THRESHOLD:

        final_result = "FORGED / TAMPERED IMAGE"
        confidence = tampered_prob
        stage = "CASIA V2.0 FORGERY DETECTOR"

    else:

        final_result = "AUTHENTIC / REAL IMAGE"

        # Use CASIA authenticity probability only.
        # Do not compare probabilities from two different models.
        confidence = authentic_prob

        stage = "CASIA V2.0 FORGERY DETECTOR"

    return {
        "final_result": final_result,
        "stage": stage,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "tampered_probability": tampered_prob,
        "authentic_probability": authentic_prob
    }


# UI — original layout and labels retained
st.title("AI Image Forensics Detector")

st.write(
    "Upload an image to detect whether it is "
    "AI-generated, forged/tampered, or authentic."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        analyze_button = st.button(
            "Analyze Image",
            type="primary"
        )

        if analyze_button:

            with st.spinner(
                "Analyzing image with both detectors..."
            ):
                result = analyze_image(image)

            st.divider()
            st.subheader("Detection Result")

            if result["final_result"] == "AI-GENERATED IMAGE":

                st.error("AI-GENERATED IMAGE")

                st.write(
                    f"Confidence: {result['confidence'] * 100:.2f}%"
                )

                st.write(
                    "Detected by: AI-Generated Image Detector"
                )

                with st.expander("Detection Details"):

                    st.write(
                        f"AI-Generated Probability: "
                        f"{result['fake_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"Real Probability: "
                        f"{result['real_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"CASIA Tampered Probability: "
                        f"{result['tampered_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"CASIA Authentic Probability: "
                        f"{result['authentic_probability'] * 100:.2f}%"
                    )

            elif result["final_result"] == "FORGED / TAMPERED IMAGE":

                st.warning("FORGED / TAMPERED IMAGE")

                st.write(
                    f"Confidence: {result['confidence'] * 100:.2f}%"
                )

                st.write(
                    "Detected by: CASIA V2.0 Forgery Detector"
                )

                with st.expander("Detection Details"):

                    st.write(
                        f"Tampered Probability: "
                        f"{result['tampered_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"Authentic Probability: "
                        f"{result['authentic_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"AI-Generated Probability: "
                        f"{result['fake_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"Real Probability: "
                        f"{result['real_probability'] * 100:.2f}%"
                    )

            else:

                st.success("AUTHENTIC / REAL IMAGE")

                st.write(
                    f"Confidence: {result['confidence'] * 100:.2f}%"
                )

                st.write(
                    "Detected by: CASIA V2.0 Forgery Detector"
                )

                with st.expander("Detection Details"):

                    st.write(
                        f"Authentic Probability: "
                        f"{result['authentic_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"Tampered Probability: "
                        f"{result['tampered_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"AI-Generated Probability: "
                        f"{result['fake_probability'] * 100:.2f}%"
                    )

                    st.write(
                        f"Real Probability: "
                        f"{result['real_probability'] * 100:.2f}%"
                    )

    except Exception as e:

        st.error(
            f"Could not process this image: {e}"
        )


st.divider()

st.caption(
    "Two-stage image analysis system using "
    "EfficientNetB0, CIFAKE and CASIA V2.0."
)

