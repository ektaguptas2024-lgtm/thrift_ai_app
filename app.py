import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# --- Page setup ---
st.set_page_config(page_title="Smart Cloth Condition Predictor 👕", layout="centered")

st.title("👕 Smart Cloth Condition Classifier")
st.write("Upload an image of clothing to predict its condition, recyclability, and potential resale value.")

# --- Load ONNX model ---
@st.cache_resource
def load_model():
    session = ort.InferenceSession("cloth_condition_model.onnx")
    return session

session = load_model()
classes = ["Torn", "Faded", "Good"]

# --- Preprocess and Predict ---
def predict_condition(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    preds = session.run(None, {"input": img_array})[0]
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return classes[pred_idx], confidence

# --- User Input ---
uploaded_image = st.file_uploader("📤 Upload a clothing image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing image...")
    label, confidence = predict_condition(image)
    st.success(f"🧾 Predicted Condition: **{label}** ({confidence*100:.1f}% confidence)")

    # --- Logical outputs based on prediction ---
    if label == "Good":
        category = "Sellable"
        price = np.random.randint(200, 500)
        co2_saved = np.random.uniform(2.5, 4.0)
        water_saved = np.random.uniform(400, 600)
        reward_points = int(price * 0.1)
    elif label == "Faded":
        category = "Recyclable"
        price = np.random.randint(50, 150)
        co2_saved = np.random.uniform(1.5, 2.5)
        water_saved = np.random.uniform(200, 400)
        reward_points = int(price * 0.15)
    else:
        category = "Recyclable"
        price = np.random.randint(10, 50)
        co2_saved = np.random.uniform(0.5, 1.5)
        water_saved = np.random.uniform(100, 200)
        reward_points = int(price * 0.2)

    # --- Display Results ---
    st.subheader("♻️ Sustainability Summary")
    st.markdown(f"""
    - **Category:** {category}  
    - **Estimated Price:** ₹{price}  
    - **CO₂ Saved:** {co2_saved:.2f} kg  
    - **Water Saved:** {water_saved:.2f} L  
    - **Reward Points:** {reward_points} pts
    """)

st.caption("Developed with 💚 for sustainable fashion.")



