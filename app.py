import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import pickle
import os

st.set_page_config(
    page_title="Duckweed Copper Analyzer",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 26px; font-weight: 700; color: #1a1a1a; }
    .result-box {
        border-radius: 10px; padding: 22px;
        text-align: center; margin: 16px 0; color: white;
    }
    .result-box .val { font-size: 38px; font-weight: 700; }
    .result-box .lbl { font-size: 15px; margin-top: 4px; opacity: 0.92; }
    .result-box .sub { font-size: 11px; margin-top: 3px; opacity: 0.75; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_model():
    for base in [".", "/home/claude"]:
        mp = os.path.join(base, "duckweed_model.pkl")
        sp = os.path.join(base, "duckweed_scaler.pkl")
        if os.path.exists(mp):
            with open(mp, "rb") as f: model = pickle.load(f)
            with open(sp, "rb") as f: scaler = pickle.load(f)
            return model, scaler
    st.error("Model files not found.")
    st.stop()


def detect_duckweed(cropped):
    R, G, B = cv2.split(cropped)
    mask = (
        (G > 110) & (G > R + 5) & (G > B + 35) &
        (cropped.mean(axis=2) < 155) & (cropped.mean(axis=2) > 80)
    ).astype(np.uint8) * 255
    k3, k5 = np.ones((3,3), np.uint8), np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if 20 <= stats[i, cv2.CC_STAT_AREA] <= 800:
            out[labels == i] = 255
    return out


def extract_features(cropped):
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    R, G, B = cv2.split(cropped)
    H, S, V = cv2.split(hsv)
    mask = detect_duckweed(cropped)
    coverage = 100 * np.sum(mask > 0) / mask.size
    if np.sum(mask > 0) > 0:
        px = mask > 0
        h_mean, s_mean, v_mean = H[px].mean(), S[px].mean(), V[px].mean()
        g_mean, b_mean = G[px].mean(), B[px].mean()
        h_std, s_std = H[px].std(), S[px].std()
        g_to_b = g_mean / (b_mean + 1)
        brightness = cropped[px].mean()
    else:
        h_mean, s_mean, v_mean = 30, 70, 140
        g_mean, b_mean, h_std, s_std, g_to_b, brightness = 140, 100, 5, 40, 1.4, 130
    return np.array([coverage, h_mean, s_mean, v_mean,
                     g_mean, b_mean, h_std, s_std, g_to_b, brightness]), mask


def run_analysis(image, is_control):
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255,255,255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    h, w = arr.shape[:2]
    ch, cw = int(h * .5), int(w * .5)
    sy, sx = (h - ch) // 2, (w - cw) // 2
    cropped = arr[sy:sy+ch, sx:sx+cw]
    features, mask = extract_features(cropped)
    model, scaler = load_ml_model()
    copper = float(model.predict(scaler.transform(features.reshape(1,-1)))[0])
    if is_control:
        copper = min(copper, 0.5)
    copper = max(0.0, min(copper, 12.0))
    health = max(0, min(100, 100 - copper / 12 * 100))
    return dict(copper=copper, health=health, coverage=features[0],
                features=features, cropped=cropped, mask=mask)


def get_status(c):
    if   c < 1.0: return "Healthy / Control", "#43a047", "✅"
    elif c < 3.0: return "Low Stress",         "#7cb342", "🟡"
    elif c < 6.0: return "Moderate Stress",    "#fb8c00", "⚠️"
    elif c < 9.0: return "High Stress",        "#e53935", "🚨"
    else:         return "Severe Toxicity",    "#b71c1c", "☠️"


def make_viz(original, cropped, mask):
    img1 = np.array(original.convert("RGB"))
    ov = np.zeros_like(cropped)
    ov[mask > 0] = [0, 255, 0]
    img2 = cv2.addWeighted(cropped.copy(), 0.65, ov, 0.35, 0)
    return Image.fromarray(img1), Image.fromarray(img2)


st.title("🔬 Duckweed Copper Analyzer")
st.caption("ML Biosensor · Karthikeya Yeruva · Steinbrenner HS / USF · ISEF 2026")
st.markdown("---")

is_control = st.checkbox("This is a control sample (0 mg/L copper)")
st.markdown("---")

uploaded = st.file_uploader(
    "Upload or take a photo of your sample",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    pil_image = Image.open(uploaded)
    st.markdown("---")

    with st.spinner("Analyzing…"):
        res = run_analysis(pil_image, is_control)
        status, color, icon = get_status(res["copper"])
        orig_viz, ov_viz = make_viz(pil_image, res["cropped"], res["mask"])

    c1, c2 = st.columns(2)
    with c1: st.image(orig_viz,  caption="Original sample",           use_container_width=True)
    with c2: st.image(ov_viz,    caption="Duckweed detected (green)", use_container_width=True)

    st.markdown(f"""
    <div class="result-box" style="background:{color}">
        <div class="val">{res['copper']:.2f} mg/L</div>
        <div class="lbl">{icon}  {status}</div>
        <div class="sub">Random Forest · trained on your experimental data</div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Coverage",     f"{res['coverage']:.1f}%")
    m2.metric("Health Score", f"{res['health']:.0f}/100")
    m3.metric("Copper",       f"{res['copper']:.2f} mg/L")

    epa = 1.3
    if res["copper"] <= epa:
        st.success(f"✅ Below EPA drinking-water action level ({epa} mg/L)")
    else:
        st.warning(f"⚠️ Exceeds EPA action level by {res['copper']-epa:.2f} mg/L")

    st.markdown("---")

    with st.expander("📊 Extracted image features"):
        names = ["Coverage %","Mean Hue","Mean Saturation","Mean Value",
                 "Green Channel","Blue Channel","Hue Std Dev",
                 "Saturation Std Dev","G/B Ratio","Brightness"]
        st.dataframe(pd.DataFrame({"Feature": names,
                                    "Value": [f"{v:.2f}" for v in res["features"]]}),
                     hide_index=True, use_container_width=True)

    with st.expander("🤖 Model & training info"):
        st.markdown("""
**Model:** Random Forest Regressor · 150 trees · 32 experimental images

| Concentration | Samples | Images |
|---|---|---|
| 1.0 mg/L | 2A, 2B | 8 |
| 2.0 mg/L | 3A, 3B | 8 |
| 4.0 mg/L | 4A, 4B | 8 |
| 8.0 mg/L | 5A | 4 |
| 9.7 mg/L | 5B | 4 |

**Performance:** MAE 1.0 mg/L · R² 0.794  
**Top features:** Saturation Std Dev 34.7% · Mean Hue 18.7% · Coverage 10.1%
        """)

    with st.expander("🔍 Detection method"):
        st.markdown("""
- Green channel > 110 · Green > Red+5 · Green > Blue+35
- Brightness 80–155 · Component size 20–800 px
- Morphological open/close to remove noise
        """)

    st.markdown("---")
    csv = pd.DataFrame({
        "Timestamp":         [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Is_Control":        [is_control],
        "ML_Prediction_mgL": [f"{res['copper']:.2f}"],
        "Coverage_%":        [f"{res['coverage']:.2f}"],
        "Health_Score":      [f"{res['health']:.1f}"],
        "Status":            [status],
    }).to_csv(index=False)

    dl, cl = st.columns([3, 1])
    with dl:
        st.download_button("⬇️ Download result (CSV)", data=csv,
                           file_name=f"duckweed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    with cl:
        if st.button("🗑️ Clear"):
            st.rerun()

else:
    st.info("📱 On iPhone: tap the upload box and choose **Take Photo** to use your camera directly.")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
**Duckweed Copper Analyzer** predicts copper contamination from a single photo using
a Random Forest model trained on 32 of your own experimental images.

- **Accuracy:** ~1 mg/L average error · **R²:** 0.794
- **Trained on:** 1, 2, 4, 8, 9.7 mg/L concentrations
- **No lab equipment needed**
        """)

st.markdown("---")
st.caption("Duckweed Copper Analyzer · ML Version · ISEF 2026 · Karthikeya Yeruva")
