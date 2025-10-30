import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import random

# ----------------------------
# Load ONNX Model
# ----------------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession("cloth_condition_model.onnx")
    return session

session = load_model()

# ----------------------------
# Helper functions
# ----------------------------
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: img_array})
    return np.argmax(preds[0])

# Example label names (you can adjust based on your model classes)
CLASS_NAMES = ["Good", "Recyclable", "Bad"]

# ----------------------------
# Recycled Product Suggestions
# ----------------------------
RECYCLED_PRODUCTS = [
    "Tote bag made from cloth scraps",
    "Rug or floor mat",
    "Cushion cover",
    "Quilt or blanket patch",
    "Handmade notebook cover",
    "Upcycled fashion accessory (bracelet, pouch)",
    "Soft toy filling",
    "Reusable cleaning cloths",
]

# ----------------------------
# Women NGOs in India
# ----------------------------
NGO_LIST = [
    {"name": "EmpowHER India", "cause": "Empowering rural women through education and livelihood."},
    {"name": "Swayam Shikshan Prayog", "cause": "Supporting women-led sustainable enterprises."},
    {"name": "Udyogini", "cause": "Building entrepreneurship among rural women."},
    {"name": "SAFA India", "cause": "Skill-building and market access for women artisans."},
    {"name": "AIWC", "cause": "Promoting women's rights and education."},
    {"name": "SEWA", "cause": "Organizing informal women workers."},
    {"name": "Goonj", "cause": "Recycling urban waste for rural development."},
    {"name": "Stree Mukti Sanghatana", "cause": "Empowering women waste pickers."},
    {"name": "Sneha", "cause": "Health and well-being for women and children."},
    {"name": "Mann Deshi Foundation", "cause": "Financial literacy for rural women entrepreneurs."},
]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("👕 Cloth Condition Detection & Recycling Suggestion App")
st.markdown("Upload a cloth image to detect its condition and get recycling options.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing cloth condition..."):
        prediction = predict_image(image)
        predicted_label = CLASS_NAMES[prediction]
        st.subheader(f"🧾 Predicted Condition: **{predicted_label}**")

    # ----------------------------
    # If the item is recyclable, show NGO suggestions and recycled product ideas
    # ----------------------------
    if predicted_label == "Recyclable":
        st.success("♻️ This cloth can be recycled! Let's make the world greener 🌍")

        # Suggest possible recycled product
        product_suggestion = random.choice(RECYCLED_PRODUCTS)
        st.markdown(f"**Suggested recycled product:** {product_suggestion}")

        # List of NGOs
        st.markdown("### 👩‍🧵 Women NGOs that could help recycle this:")
        for ngo in NGO_LIST:
            st.markdown(f"**{ngo['name']}** — _{ngo['cause']}_")

        # Choose NGO
        selected_ngo = st.selectbox("Select an NGO to connect with:", [ngo["name"] for ngo in NGO_LIST])
        st.info(f"You selected: {selected_ngo}")

        # Choose whether to buy or sell
        st.markdown("### 💰 What would you like to do next?")
        action = st.radio("Choose your option:", ["Buy recycled product", "Sell recycled product"])

        if action == "Buy recycled product":
            st.success(f"🛍 You’ve chosen to **buy** a {product_suggestion}. Contact **{selected_ngo}** for details.")
        elif action == "Sell recycled product":
            st.success(f"💼 You’ve chosen to **sell** a {product_suggestion}. **{selected_ngo}** can assist in connecting to buyers.")
    else:
        st.info("This cloth is not marked recyclable. You may reuse or donate it if possible.")

st.markdown("---")
st.caption("Developed with ❤️ for sustainable fashion and women empowerment.")
