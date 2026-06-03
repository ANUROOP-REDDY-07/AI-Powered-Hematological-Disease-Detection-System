import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hematology AI",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center;'>🩸 Hematology Disease Detection</h1>
<p style='text-align:center;'>AI-powered Blood Cell Analysis with Explainable AI (Grad-CAM)</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "C:\Projects\hematology_app\hematology_final_model_14.keras",
        compile=False
    )

model = load_model()

class_names = ['leukemia', 'malaria', 'normal', 'sickle']

# =========================
# DISEASE INFO
# =========================
disease_info = {
    "leukemia": "Cancer of blood-forming tissues affecting white blood cells.",
    "malaria": "Parasitic infection transmitted by mosquitoes.",
    "normal": "Healthy blood cells with no abnormalities.",
    "sickle": "Genetic disorder causing abnormal crescent-shaped red blood cells."
}

# =========================
# FIND LAST CONV LAYER
# =========================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

# =========================
# GRAD-CAM FUNCTION
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload Blood Cell Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # =========================
    # IMAGE PROCESSING
    # =========================
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # =========================
    # PREDICTION
    # =========================
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    # =========================
    # GRAD-CAM
    # =========================
    last_conv_layer_name = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img.copy(), 0.7, heatmap, 0.3, 0)

    # =========================
    # CENTERED IMAGE DISPLAY
    # =========================
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

    st.image(img, caption="Original Image", width="stretch")
    st.image(overlay, caption="Grad-CAM Visualization", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # PREDICTION CARD
    # =========================
    st.markdown(f"""
    <div style='text-align:center; padding:20px; background-color:#111; border-radius:12px; margin-top:20px;'>
        <h2 style='color:#00FFAA;'>Prediction: {pred_class.upper()}</h2>
        <h3 style='color:#FFD700;'>Confidence: {confidence:.2%}</h3>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # CONFIDENCE BAR
    # =========================
    st.progress(int(confidence * 100))

    # =========================
    # PROBABILITY CHART
    # =========================
    st.subheader("📊 Class Probabilities")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": preds
    })

    st.bar_chart(prob_df.set_index("Class"))

    # =========================
    # DISEASE INFO
    # =========================
    st.subheader("🧬 Disease Information")
    st.info(disease_info[pred_class])

    # =========================
    # DOWNLOAD REPORT
    # =========================
    report_text = f"""
Prediction: {pred_class}
Confidence: {confidence:.2%}
"""

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="hematology_report.txt"
    )