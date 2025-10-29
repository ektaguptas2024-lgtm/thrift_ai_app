import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ------------------------------
# Load TFLite Model
# ------------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="cloth_condition_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Preprocess image
# ------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# ------------------------------
# Predict function
# ------------------------------
def predict_condition(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])
    class_names = ['Torn', 'Faded', 'Good']
    return class_names[predicted_class], output_data[0]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("♻️ Cloth Condition Classifier (Lite)")
st.write("Upload a cloth image to predict whether it’s **Torn**, **Faded**, or **Good**.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        predicted_label, confidence = predict_condition(image)

    st.success(f"🧵 Predicted: **{predicted_label}**")

    # Optional sustainability metrics
    details = {
        "Good":  ("Can be sold directly.", 150, 2.5, 300, 50),
        "Faded": ("Could be upcycled.", 75, 1.5, 150, 30),
        "Torn":  ("Best for recycling.", 30, 0.8, 80, 15)
    }
    remark, price, co2, water, points = details[predicted_label]

    st.write(f"💰 **Price:** ₹{price}")
    st.write(f"🌍 **CO₂ Saved:** {co2} kg")
    st.write(f"💧 **Water Saved:** {water} L")
    st.write(f"🏅 **Reward Points:** {points}")
    st.info(remark)


