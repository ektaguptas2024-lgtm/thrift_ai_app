# app.py — ReWear Thrift AI with NGO & Upcycling Suggestions
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

st.set_page_config(page_title="ReWear — Thrift AI Demo", layout="centered")

# ---------- Helper functions ----------

def read_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return img

def auto_analyze_image_pil(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges) / edges.size)
    color_var = float(np.mean(np.var(img / 255.0, axis=(0,1))))
    return {"brightness": brightness, "edge_density": edge_density, "color_var": color_var}

def decide_torn_faded(auto_metrics, user_override=None):
    b = auto_metrics["brightness"]
    e = auto_metrics["edge_density"]
    c = auto_metrics["color_var"]

    torn_score = e * 10
    faded_score = (1 - c) * (255 - abs(b - 128)) / 128

    torn = True if torn_score > 0.8 else False
    faded = True if faded_score > 1.2 else False

    if user_override in ("Yes", "No"):
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
    age_factor = max(0.15, 1 - 0.12 * age_years)
    damage_factor = 1.0
    if torn:
        damage_factor *= 0.35
    if faded:
        damage_factor *= 0.7
    brand_factor = 1.5 if branded == "Yes" else 1.0
    if condition_factor:
        damage_factor *= condition_factor

    price = base_price * brand_factor * age_factor * damage_factor
    price = max(round(price, 2), 10.0)
    return price

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

# ---------- NEW FEATURE: NGO & Upcycling Suggestion ----------
def suggest_recycling_options(item_type):
    suggestions = {
        "T-shirt/top": {
            "NGO": "Usha Silai Women’s Collective, Delhi",
            "UpcycleIdea": "Soft cleaning rags or patchwork quilt"
        },
        "Trouser": {
            "NGO": "SEWA Mahila Cooperative, Ahmedabad",
            "UpcycleIdea": "Stylish handbags or reusable grocery bags"
        },
        "Dress": {
            "NGO": "Goonj Foundation, Delhi",
            "UpcycleIdea": "Reusable fabric pouches or aprons"
        },
        "Bag": {
            "NGO": "GreenSole Women Recycle Unit, Mumbai",
            "UpcycleIdea": "New small clutches or wallets"
        },
        "Coat": {
            "NGO": "Aavaran NGO, Udaipur",
            "UpcycleIdea": "Warm blankets for shelters"
        },
        "Shirt": {
            "NGO": "Urmul Trust, Rajasthan",
            "UpcycleIdea": "Children’s school uniforms"
        },
        "Other": {
            "NGO": "Goonj Foundation, Delhi",
            "UpcycleIdea": "General recycled fabric craft items"
        }
    }
    return suggestions.get(item_type, suggestions["Other"])

# ---------- App UI ----------

st.title("ReWear — Smart Thrift AI (Demo)")
st.caption("Demo uses lightweight heuristics for image condition detection. Replace with real model API later.")
st.markdown("---")

st.header("1. Upload item photo")
uploaded_file = st.file_uploader("Upload clothing image (jpg/png)", type=["jpg","jpeg","png"])

predicted_category = "Other"

if uploaded_file:
    pil_img = read_image(uploaded_file)
    st.image(pil_img, caption="Uploaded image", use_column_width=True)
    metrics = auto_analyze_image_pil(pil_img)
    st.write(f"Brightness: {metrics['brightness']:.1f} | Edge density: {metrics['edge_density']:.4f} | Color variance: {metrics['color_var']:.4f}")

    w, h = pil_img.size
    aspect = w / h
    arr = np.array(pil_img) / 255.0
    colorfulness = float(np.mean(np.std(arr, axis=(0,1))))
    if colorfulness > 0.25 and aspect < 1.2:
        predicted_category = "Dress"
    elif aspect > 1.6:
        predicted_category = "T-shirt/top"
    elif aspect > 1.0 and aspect <= 1.6:
        predicted_category = "Shirt"

    st.subheader(f"Auto-detected item type: **{predicted_category}**")

st.markdown("---")
st.header("2. Manual details")
col1, col2 = st.columns(2)
with col1:
    item_type = st.selectbox("Final Item Type", [predicted_category,"T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot","Other"], index=0)
    branded = st.selectbox("Branded?", ["Yes","No"])
    age = st.slider("Age of item (years)", 0, 15, 1)
with col2:
    user_torn = st.selectbox("Torn (override)?", ["Auto Detect","Yes","No"])
    user_faded = st.selectbox("Faded (override)?", ["Auto Detect","Yes","No"])
    manual_condition_factor = st.slider("Manual condition multiplier", 0.2, 1.5, 1.0, 0.05)

st.markdown("---")
st.header("3. Analyze & Compute")

if st.button("Run Analysis"):
    if not uploaded_file:
        st.error("Please upload an image first.")
    else:
        auto_metrics = auto_analyze_image_pil(read_image(uploaded_file))
        torn_auto, faded_auto, reason = decide_torn_faded(auto_metrics, user_override=(user_torn if user_torn!="Auto Detect" else None))
        faded = (user_faded == "Yes") if user_faded != "Auto Detect" else faded_auto
        torn = torn_auto if user_torn == "Auto Detect" else (user_torn == "Yes")

        st.write("### Condition Analysis")
        st.write(f"- Torn: **{'Yes' if torn else 'No'}**")
        st.write(f"- Faded: **{'Yes' if faded else 'No'}**")
        st.write(f"- Reason: {reason}")

        if torn or (faded and age > 7):
            decision = "Recyclable"
        else:
            decision = "Sellable"

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
        water_saved, co2_saved = sustainability_stats(item_type, decision)
        reward_pts = reward_points_for_seller(decision, final_price)

        if decision == "Sellable":
            st.success("🛍️ Decision: **Sellable** — Good for resale")
        else:
            st.warning("♻️ Decision: **Recyclable** — Recommend recycling / upcycling")
            recycle_info = suggest_recycling_options(item_type)
            st.markdown("### ♻️ Recycling Options")
            st.write(f"**Recommended NGO:** {recycle_info['NGO']}")
            st.write(f"**Possible upcycled product:** {recycle_info['UpcycleIdea']}")

            seller_choice = st.radio(
                "Would you like to:",
                ["Buy recycled product yourself", "Sell the recycled product via platform"]
            )

            if seller_choice == "Buy recycled product yourself":
                st.success("You chose to buy the upcycled product. NGO will contact you soon.")
            else:
                st.info("You chose to sell the recycled product on ReWear Marketplace.")

        st.metric("Estimated Resale Price (INR)", f"₹{final_price:.2f}")
        st.write(f"💧 Water saved: **{water_saved} L**")
        st.write(f"🌤️ CO₂ avoided: **{co2_saved} kg**")
        st.write(f"🏆 Reward points: **{reward_pts} pts**")

        st.write("---")
        st.json({
            "timestamp": datetime.utcnow().isoformat(),
            "item_type": item_type,
            "final_decision": decision,
            "estimated_price_inr": final_price,
            "seller_reward_points": reward_pts
        })

        if st.button("Add to Marketplace (simulate)"):
            st.success("Item added to marketplace simulation. Reward points credited!")

st.markdown("---")
st.caption("This demo uses heuristic analysis for image condition. Replace with your trained model API when ready.")
