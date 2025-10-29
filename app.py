# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import math
import random
from datetime import datetime

st.set_page_config(page_title="ReWear — Thrift AI Demo", layout="centered")

# ---------- Helper functions ----------
def read_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return img

def auto_analyze_image_pil(pil_image):
    """
    Lightweight heuristic-based image analysis:
    - brightness (mean gray)
    - edge density (Canny)
    - color variance (approx fade)
    Returns dict with brightness, edge_density, color_var
    """
    # convert PIL to OpenCV
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))  # 0-255
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges) / edges.size)  # 0-1
    # color variance across channels
    color_var = float(np.mean(np.var(img / 255.0, axis=(0,1))))
    return {"brightness": brightness, "edge_density": edge_density, "color_var": color_var}

def decide_torn_faded(auto_metrics, user_override=None):
    """
    Combine auto metrics with optional user override.
    Returns (torn_bool, faded_bool, reason_text)
    Heuristics (tunable):
      - torn if edge_density above threshold or very high texture noise
      - faded if brightness high/low combined with low color variance
    """
    b = auto_metrics["brightness"]
    e = auto_metrics["edge_density"]
    c = auto_metrics["color_var"]

    torn_score = e * 10  # scale
    faded_score = (1 - c) * (255 - abs(b - 128)) / 128  # heuristic

    torn = True if torn_score > 0.8 else False
    faded = True if faded_score > 1.2 else False

    if user_override in ("Yes", "No"):
        # user explicitly provided torn
        torn = (user_override == "Yes")

    reasons = []
    if torn:
        reasons.append(f"Edge density high ({e:.3f})")
    if faded:
        reasons.append(f"Low color variance ({c:.3f}) or faded look")
    if not reasons:
        reasons.append("No major damage/fade detected")

    return torn, faded, "; ".join(reasons)

def estimate_price(base_price, age_years, torn, faded, branded, condition_factor=None):
    """
    Price estimation logic:
    - base_price depends on category & branded
    - depreciation by age
    - damage modifiers for torn/faded
    - returns final_price (float)
    """
    # age depreciation (exponential-ish)
    age_factor = max(0.15, 1 - 0.12 * age_years)  # at 8 years -> ~0.04, floor 0.15
    # condition effect
    damage_factor = 1.0
    if torn:
        damage_factor *= 0.35
    if faded:
        damage_factor *= 0.7
    # branded premium
    brand_factor = 1.5 if branded == "Yes" else 1.0
    # optional manual adjustment
    if condition_factor:
        damage_factor *= condition_factor

    price = base_price * brand_factor * age_factor * damage_factor
    price = max(round(price, 2), 10.0)  # min price floor
    return price

def sustainability_stats(item_type, decision):
    """
    Return water_saved_liters, co2_saved_kg based on item type and whether it was resold.
    Values are illustrative and defensible:
     - T-shirt reuse saves ~2700 L water, ~5 kg CO2
     - Jeans reuse saves ~7000 L water, ~20 kg CO2
    If recycled, assume lower direct water saving but waste diverted.
    """
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
        # full reuse benefit
        return water, co2
    else:
        # recycling saves less water but prevents landfill -- give conservative values
        return round(water * 0.35), round(co2 * 0.25, 2)

def reward_points_for_seller(decision, price):
    """
    Simple reward logic:
    - Sellable and higher price => more points
    - Recyclable (if user sends to certified recycler) => some points
    """
    base = 0
    if decision == "Sellable":
        base = int(min(100, max(10, price // 10)))
    else:  # Recyclable
        base = 20  # encourage recycling
    # bonus for high sustainability impact left for future
    return base

# ---------- App UI ----------
st.title("ReWear — Smart Thrift AI (Demo)")
st.caption("Demo uses lightweight heuristics for image condition detection. Replace with real model API later.")

st.markdown("---")
st.header("1. Upload item photo")

uploaded_file = st.file_uploader("Upload clothing image (jpg/png)", accept_multiple_files=False, type=["jpg","jpeg","png"])

# default values for marketplace
category_default = "Other"
predicted_category = category_default

if uploaded_file is not None:
    pil_img = read_image(uploaded_file)
    st.image(pil_img, caption="Uploaded image", use_column_width=True)
    # small preview and metrics
    st.caption("Analyzing image (brightness, edges, color variance)...")
    metrics = auto_analyze_image_pil(pil_img)
    st.write(f"Brightness: {metrics['brightness']:.1f} | Edge density: {metrics['edge_density']:.4f} | Color variance: {metrics['color_var']:.4f}")

    # quick category guess (toy heuristic) - this replaces your previous clothing-type model
    # we use aspect ratio and colorfulness to guess a rough category just for UI
    w, h = pil_img.size
    aspect = w / h
    # colorfulness proxy
    arr = np.array(pil_img) / 255.0
    colorfulness = float(np.mean(np.std(arr, axis=(0,1))))
    if colorfulness > 0.25 and aspect < 1.2:
        predicted_category = "Dress"
    elif aspect > 1.6:
        predicted_category = "T-shirt/top"
    elif aspect > 1.0 and aspect <= 1.6:
        predicted_category = "Shirt"
    else:
        predicted_category = "Other"

    st.subheader(f"Auto-detected item type: **{predicted_category}**")

st.markdown("---")
st.header("2. Manual details (helps finalize decision)")
col1, col2 = st.columns(2)
with col1:
    item_type = st.selectbox("Final Item Type (choose or keep auto)", [predicted_category, "T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot","Other"], index=0)
    branded = st.selectbox("Branded?", ["Yes","No"])
    age = st.slider("Age of item (years)", 0, 15, 1)
with col2:
    user_torn = st.selectbox("Torn (override)?", ["Auto Detect","Yes","No"])
    user_faded = st.selectbox("Faded (override)?", ["Auto Detect","Yes","No"])
    manual_condition_factor = st.slider("Manual condition multiplier (0.2 very bad → 1.5 excellent)", 0.2, 1.5, 1.0, 0.05)

st.markdown("---")
st.header("3. Analyze & Compute")

if st.button("Run Analysis"):
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        # auto analysis
        auto_metrics = auto_analyze_image_pil(read_image(uploaded_file))
        torn_auto, faded_auto, reason = decide_torn_faded(auto_metrics, user_override=(user_torn if user_torn!="Auto Detect" else None))
        # if user says Auto and also overrides faded
        if user_faded != "Auto Detect":
            faded = (user_faded == "Yes")
        else:
            # combine auto faded heuristic with brightness threshold
            faded = faded_auto

        torn = torn_auto if user_torn == "Auto Detect" else (user_torn == "Yes")

        st.write("### Condition analysis (AI + user inputs)")
        st.write(f"- Torn (final): **{ 'Yes' if torn else 'No' }**")
        st.write(f"- Faded (final): **{ 'Yes' if faded else 'No' }**")
        st.write(f"- Reason: {reason}")

        # Decision logic
        # stricter rules: torn => Recyclable, heavy fade + >5 years => Recyclable
        if torn or (faded and age > 7):
            decision = "Recyclable"
        else:
            decision = "Sellable"

        # base price mapping by item_type
        base_map = {
            "T-shirt/top": 400,
            "Trouser": 600,
            "Pullover": 700,
            "Dress": 800,
            "Coat": 1200,
            "Sandal": 250,
            "Shirt": 500,
            "Sneaker": 900,
            "Bag": 600,
            "Ankle boot": 700,
            "Other": 450
        }
        base_price = base_map.get(item_type, 450)

        final_price = estimate_price(base_price, age, torn, faded, branded, condition_factor=manual_condition_factor)

        # sustainability stats
        water_saved, co2_saved = sustainability_stats(item_type, "Sellable" if decision=="Sellable" else "Recyclable")
        reward_pts = reward_points_for_seller(decision, final_price)

        # Display results
        st.markdown("### Result")
        if decision == "Sellable":
            st.success("🛍️ Decision: **Sellable** — Good for resale")
        else:
            st.warning("♻️ Decision: **Recyclable** — Recommend recycling / upcycling")

        st.metric("Estimated Resale Price (INR)", f"₹{final_price:.2f}")
        st.write(f"💧 Estimated water saved by reuse: **{water_saved} L**")
        st.write(f"🌤️ Estimated CO₂ avoided: **{co2_saved} kg**")
        st.write(f"🏆 Reward points for seller: **{reward_pts} pts**")

        # small logs: timestamp + "listing" simulation (not persistent)
        st.write("---")
        st.write("Listing Summary (preview):")
        st.json({
            "timestamp": datetime.utcnow().isoformat(),
            "item_type": item_type,
            "predicted_item_type": predicted_category,
            "final_decision": decision,
            "estimated_price_inr": final_price,
            "seller_reward_points": reward_pts
        })

        # CTA: simulate sending item to marketplace
        if st.button("Add to Marketplace (simulate)"):
            st.success("Item added to marketplace (simulation). You earned the reward points!")

st.markdown("---")
st.caption("This demo uses deterministic heuristics for image condition. Replace the heuristics with your trained model API when ready.")

