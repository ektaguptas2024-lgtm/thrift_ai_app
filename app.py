# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
import math
import random
from datetime import datetime

st.set_page_config(page_title="ReWear — Thrift AI Demo", layout="centered")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("cloth_condition_model.keras")
        st.success("✅ AI Model Loaded Successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}. Using fallback heuristic detection.")
        return None

model = load_model()

# ---------- Helper functions ----------
def read_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return img

def auto_analyze_image_pil(pil_image):
    """Heuristic fallback if no model is loaded"""
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges) / edges.size)
    color_var = float(np.mean(np.var(img / 255.0, axis=(0,1))))
    return {"brightness": brightness, "edge_density": edge_density, "color_var": color_var}

def predict_with_model(pil_image):
    """AI Model Prediction"""
    img = pil_image.resize((128, 128))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]

    # Assuming model outputs [torn_prob, faded_prob, category_logits...]
    torn = preds[0] > 0.5
    faded = preds[1] > 0.5
    categories = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Shirt","Sneaker","Bag","Other"]
    category_pred = categories[int(np.argmax(preds[2:]))] if len(preds) > 2 else "Other"
    return torn, faded, category_pred

def decide_torn_faded(auto_metrics):
    """Heuristic logic"""
    b = auto_metrics["brightness"]
    e = auto_metrics["edge_density"]
    c = auto_metrics["color_var"]

    torn_score = e * 10
    faded_score = (1 - c) * (255 - abs(b - 128)) / 128

    torn = torn_score > 0.8
    faded = faded_score > 1.2

    reasons = []
    if torn:
        reasons.append(f"High edge density ({e:.3f})")
    if faded:
        reasons.append(f"Low color variance ({c:.3f})")
    if not reasons:
        reasons.append("Good overall condition")

    return torn, faded, "; ".join(reasons)

def estimate_price(base_price, age_years, torn, faded, branded, condition_factor=1.0):
    age_factor = max(0.15, 1 - 0.12 * age_years)
    damage_factor = 1.0
    if torn: damage_factor *= 0.35
    if faded: damage_factor *= 0.7
    brand_factor = 1.5 if branded == "Yes" else 1.0
    price = base_price * brand_factor * age_factor * damage_factor * condition_factor
    return max(round(price, 2), 10.0)

def sustainability_stats(item_type, decision):
    lookup = {
        "T-shirt/top": (2700, 5),
        "Trouser": (4000, 8),
        "Pullover": (3000, 6),
        "Dress": (3500, 7),
        "Coat": (4500, 9),
        "Sandal": (200, 0.5),
        "Shirt": (2500, 4.5),
        "Sneaker": (1500, 3),
        "Bag": (1000, 2),
        "Ankle boot": (1200, 2.5),
        "Other": (2000, 4)
    }
    water, co2 = lookup.get(item_type, lookup["Other"])
    if decision == "Sellable":
        return water, co2
    else:
        return round(water * 0.35), round(co2 * 0.25, 2)

def reward_points_for_seller(decision, price):
    if decision == "Sellable":
        return int(min(100, max(10, price // 10)))
    else:
        return 20

# ---------- UI ----------
st.title("👕 ReWear — Smart Thrift AI")
st.caption("Automatically detect torn/faded clothes, estimate resale price, CO₂ savings, and reward points.")

uploaded_file = st.file_uploader("Upload clothing image (jpg/png)", type=["jpg","jpeg","png"])

predicted_category = "Other"
torn = faded = False

if uploaded_file:
    pil_img = read_image(uploaded_file)
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if model:
        torn, faded, predicted_category = predict_with_model(pil_img)
        reason = "Predicted by AI Model"
    else:
        metrics = auto_analyze_image_pil(pil_img)
        torn, faded, reason = decide_torn_faded(metrics)

    st.write(f"🧵 Torn: {'Yes' if torn else 'No'} | 🎨 Faded: {'Yes' if faded else 'No'}")
    st.caption(f"Reason: {reason}")
    st.write(f"Auto-detected category: **{predicted_category}**")

st.markdown("---")
st.header("Manual details (optional)")
col1, col2 = st.columns(2)
with col1:
    item_type = st.selectbox("Item Type", ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot","Other"], index=0)
    branded = st.selectbox("Branded?", ["Yes","No"])
    age = st.slider("Age (years)", 0, 15, 1)
with col2:
    manual_condition_factor = st.slider("Condition multiplier", 0.2, 1.5, 1.0, 0.05)

if st.button("Analyze & Estimate"):
    if not uploaded_file:
        st.error("Please upload an image first!")
    else:
        decision = "Recyclable" if torn or (faded and age > 7) else "Sellable"
        base_price = {
            "T-shirt/top": 400, "Trouser": 600, "Pullover": 700, "Dress": 800,
            "Coat": 1200, "Sandal": 250, "Shirt": 500, "Sneaker": 900,
            "Bag": 600, "Ankle boot": 700, "Other": 450
        }.get(item_type, 450)
        final_price = estimate_price(base_price, age, torn, faded, branded, manual_condition_factor)
        water_saved, co2_saved = sustainability_stats(item_type, decision)
        reward_pts = reward_points_for_seller(decision, final_price)

        st.markdown("### 💡 Result Summary")
        if decision == "Sellable":
            st.success("🛍️ Decision: Sellable — Good condition for resale!")
        else:
            st.warning("♻️ Decision: Recyclable — Recommend recycling/upcycling.")
        st.metric("Estimated Price", f"₹{final_price:.2f}")
        st.write(f"💧 Water Saved: **{water_saved} L**")
        st.write(f"🌍 CO₂ Saved: **{co2_saved} kg**")
        st.write(f"🏆 Reward Points: **{reward_pts} pts**")

        st.write("---")
        st.json({
            "timestamp": datetime.utcnow().isoformat(),
            "item_type": item_type,
            "predicted_item_type": predicted_category,
            "decision": decision,
            "price_inr": final_price,
            "reward_points": reward_pts
        })

