import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Crop Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "crop_disease_model.keras",
        compile=False
    )


# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as file:
        return json.load(file)


model = load_model()
class_names = load_class_names()


# --------------------------------------------------
# MAKE CLASS NAME EASY TO READ
# --------------------------------------------------

def clean_class_name(name):
    name = name.replace("___", " - ")
    name = name.replace("__", " - ")
    name = name.replace("_", " ")
    return name


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_disease(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image to the size used during training
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image).astype(np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:

        class_name = class_names[index]

        confidence = float(
            predictions[index] * 100
        )

        results.append(
            (class_name, confidence)
        )

    return results


# --------------------------------------------------
# WEBSITE TITLE
# --------------------------------------------------

st.title("🌿 AI-Based Crop Disease Detector")

st.write(
    "Upload a crop leaf image and the AI model "
    "will predict the most likely disease."
)


# --------------------------------------------------
# INFORMATION MESSAGE
# --------------------------------------------------

st.info(
    "⚠️ This is an educational AI decision-support "
    "project. For real agricultural decisions, "
    "confirm the result with a qualified agricultural expert."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📷 Upload Crop Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Crop Leaf",
        width="stretch"
    )

    # Prediction button
    if st.button(
        "🔍 Detect Disease",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "🤖 AI is analyzing the leaf..."
        ):

            results = predict_disease(image)

        # Best prediction
        predicted_class = results[0][0]

        confidence = results[0][1]

        readable_name = clean_class_name(
            predicted_class
        )


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.subheader("🌱 Prediction Result")

        st.success(
            f"Prediction: {readable_name}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )


        # --------------------------------------------------
        # CONFIDENCE MESSAGE
        # --------------------------------------------------

        if confidence >= 80:

            st.success(
                "✅ The model has relatively high "
                "confidence in this prediction."
            )

        elif confidence >= 60:

            st.warning(
                "⚠️ The model has moderate confidence. "
                "Try uploading a clearer image of the leaf."
            )

        else:

            st.warning(
                "⚠️ The model has low confidence. "
                "Please upload a clear leaf image and "
                "confirm the result with an agricultural expert."
            )


        # --------------------------------------------------
        # TOP 3 PREDICTIONS
        # --------------------------------------------------

        st.subheader("📊 Top 3 Predictions")

        for number, (class_name, probability) in enumerate(
            results,
            start=1
        ):

            readable_class = clean_class_name(
                class_name
            )

            st.write(
                f"**{number}. {readable_class}**"
            )

            st.progress(
                min(probability / 100, 1.0)
            )

            st.caption(
                f"Confidence: {probability:.2f}%"
            )


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    """
    This project uses Artificial Intelligence and
    Deep Learning to classify crop leaf diseases.

    Model: MobileNetV2

    Dataset: PlantVillage

    Input: Crop leaf image

    Output: Predicted crop disease and confidence score
    """
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Crop Disease Detector | "
    "MobileNetV2 + PlantVillage Dataset"
)