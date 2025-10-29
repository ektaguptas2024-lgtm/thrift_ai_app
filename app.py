import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model("cloth_condition_model.h5")
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("♻️ Smart Thrift Store AI Classifier")

uploaded_file = st.file_uploader("Upload an image of the clothing", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = class_labels[np.argmax(pred)]

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"🧥 Predicted category: **{pred_class}**")

    # --- User inputs for condition ---
    age = st.number_input("How old is the cloth (in years)?", min_value=0, max_value=20)
    torn = st.selectbox("Is it torn?", ["No", "Yes"])
    faded = st.selectbox("Is it faded?", ["No", "Yes"])
    branded = st.selectbox("Is it branded?", ["Yes", "No"])

    # --- Rule-based decision ---
    if torn == "Yes" or faded == "Yes" or age > 3:
        decision = "♻️ Recyclable"
    else:
        decision = "🛍️ Sellable"

    # --- Pricing logic (optional) ---
    base_price = 500 if branded == "Yes" else 200
    if decision == "Recyclable":
        price = base_price * 0.4
    else:
        price = base_price

    st.success(f"✅ Recommended Action: **{decision}**")
    st.info(f"💰 Suggested Price: ₹{price:.2f}")
