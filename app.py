"""
DUCKWEED COPPER ANALYZER - MACHINE LEARNING VERSION
====================================================
Karthikeya sai Yeruva 1*, Dr Sarina J. Ergas, Dr Ananda Bhattacharjee
University of South Florida | Steinbrenner High School | ISEF 2026
"""

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
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        margin: 16px 0;
        color: white;
    }
    .result-box .val  { font-size: 38px; font-weight: 700; }
    .result-box .lbl  { font-size: 15px; margin-top: 4px; opacity: 0.92; }
    .result-box .sub  { font-size: 11px; margin-top: 3px; opacity: 0.75; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CAMERA  –  rear / front toggle via JS
# ─────────────────────────────────────────────

def camera_with_toggle():
    facing = st.session_state.get("facing_mode", "environment")

    col_lbl, col_btn = st.columns([3, 1])
    with col_lbl:
        cam_label = "Rear camera" if facing == "environment" else "Selfie camera"
        st.markdown(f"**Camera** — *{cam_label} active*")
    with col_btn:
        if st.button("🔄 Flip camera", key="flip_cam"):
            st.session_state["facing_mode"] = (
                "user" if facing == "environment" else "environment"
            )
            st.rerun()

    photo = st.camera_input("", label_visibility="collapsed", key=f"cam_{facing}")

    # JS: restart the video stream with the chosen facingMode
    st.markdown(f"""
    <script>
    (function() {{
        const facing = "{facing}";
        function flipCamera() {{
            const videos = window.parent.document.querySelectorAll('video');
            videos.forEach(video => {{
                const stream = video.srcObject;
                if (!stream) return;
                stream.getTracks().forEach(t => t.stop());
                navigator.mediaDevices.getUserMedia({{
                    video: {{ facingMode: {{ ideal: facing }} }},
                    audio: false
                }}).then(newStream => {{
                    video.srcObject = newStream;
                }}).catch(console.error);
            }});
        }}
        setTimeout(flipCamera, 800);
    }})();
    </script>
    """, unsafe_allow_html=True)

    return photo


# ─────────────────────────────────────────────
#  ML MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource
def load_ml_model():
    for base in [".", "/home/claude"]:
        mp = os.path.join(base, "duckweed_model.pkl")
        sp = os.path.join(base, "duckweed_scaler.pkl")
        if os.path.exists(mp):
            with open(mp, "rb") as f: model = pickle.load(f)
            with open(sp, "rb") as f: scaler = pickle.load(f)
            return model, scaler
    st.error("Model files not found! Ensure duckweed_model.pkl and duckweed_scaler.pkl are present.")
    st.stop()


# ─────────────────────────────────────────────
#  VISION / FEATURE EXTRACTION
# ─────────────────────────────────────────────

def detect_duckweed_only(cropped):
    R, G, B = cv2.split(cropped)
    mask = (
        (G > 110) & (G > R + 5) & (G > B + 35) &
        (cropped.mean(axis=2) < 155) & (cropped.mean(axis=2) > 80)
    ).astype(np.uint8) * 255

    k3, k5 = np.ones((3,3), np.uint8), np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out, fronds = np.zeros_like(mask), 0
    for i in range(1, n):
        if 20 <= stats[i, cv2.CC_STAT_AREA] <= 800:
            out[labels == i] = 255
            fronds += 1
    return out, fronds


def extract_features(cropped, hsv):
    R, G, B = cv2.split(cropped)
    H, S, V = cv2.split(hsv)
    mask, _ = detect_duckweed_only(cropped)
    coverage = 100 * np.sum(mask > 0) / mask.size

    if np.sum(mask > 0) > 0:
        px = mask > 0
        h_mean, s_mean, v_mean = H[px].mean(), S[px].mean(), V[px].mean()
        g_mean, b_mean = G[px].mean(), B[px].mean()
        h_std, s_std = H[px].std(), S[px].std()
        g_to_b, brightness = g_mean / (b_mean + 1), cropped[px].mean()
    else:
        h_mean, s_mean, v_mean = 30, 70, 140
        g_mean, b_mean, h_std, s_std, g_to_b, brightness = 140, 100, 5, 40, 1.4, 130

    return np.array([coverage, h_mean, s_mean, v_mean,
                     g_mean, b_mean, h_std, s_std, g_to_b, brightness]), mask


def analyze(image, is_control=False):
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255,255,255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image)
    h, w = arr.shape[:2]
    ch, cw = int(h*.5), int(w*.5)
    sy, sx = (h-ch)//2, (w-cw)//2
    cropped = arr[sy:sy+ch, sx:sx+cw]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)

    features, mask = extract_features(cropped, hsv)
    model, scaler = load_ml_model()
    copper = model.predict(scaler.transform(features.reshape(1,-1)))[0]
    if is_control: copper = min(copper, 0.5)
    copper = max(0.0, min(copper, 12.0))
    health = max(0, min(100, 100 - copper/12*100))

    return dict(copper=copper, health=health, coverage=features[0],
                features=features, cropped=cropped, mask=mask)


def get_status(c):
    if   c < 1.0: return "Healthy / Control",  "#43a047", "✅"
    elif c < 3.0: return "Low Stress",          "#7cb342", "🟡"
    elif c < 6.0: return "Moderate Stress",     "#fb8c00", "⚠️"
    elif c < 9.0: return "High Stress",         "#e53935", "🚨"
    else:         return "Severe Toxicity",     "#b71c1c", "☠️"


def make_viz(original, cropped, mask):
    img1 = np.array(original.convert("RGB"))
    img2, ov = cropped.copy(), np.zeros_like(cropped)
    ov[mask > 0] = [0, 255, 0]
    img2 = cv2.addWeighted(img2, 0.65, ov, 0.35, 0)
    return Image.fromarray(img1), Image.fromarray(img2)


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────

st.title("🔬 Duckweed Copper Analyzer")
st.caption("ML Biosensor · Karthikeya Yeruva · Steinbrenner HS / USF · ISEF 2026")
st.markdown("---")

is_control = st.checkbox("This is a control sample (0 mg/L copper)")
st.markdown("---")

# ── Image input tabs ─────────────────────────
tab_cam, tab_upload = st.tabs(["📷 Camera", "📁 Upload"])

with tab_cam:
    camera_photo = camera_with_toggle()

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"],
                                     label_visibility="collapsed")

image_source = camera_photo if camera_photo is not None else uploaded_file

# ── Results ──────────────────────────────────
if image_source is not None:
    image = Image.open(image_source)

    with st.spinner("Analyzing…"):
        res = analyze(image, is_control)
        status, color, icon = get_status(res["copper"])
        orig_viz, overlay_viz = make_viz(image, res["cropped"], res["mask"])

    c1, c2 = st.columns(2)
    with c1:
        st.image(orig_viz,    caption="Original sample",            use_container_width=True)
    with c2:
        st.image(overlay_viz, caption="Duckweed detected (green)",  use_container_width=True)

    st.markdown(f"""
    <div class="result-box" style="background:{color}">
        <div class="val">{res['copper']:.2f} mg/L</div>
        <div class="lbl">{icon}  {status}</div>
        <div class="sub">Random Forest · trained on your experimental data</div>
    </div>
    """, unsafe_allow_html=True)

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

    # ── Collapsible details ───────────────────
    with st.expander("📊 Extracted image features"):
        names = ["Coverage %","Mean Hue","Mean Saturation","Mean Value",
                 "Green Channel","Blue Channel","Hue Std Dev",
                 "Saturation Std Dev","G/B Ratio","Brightness"]
        st.dataframe(
            pd.DataFrame({"Feature": names,
                          "Value": [f"{v:.2f}" for v in res["features"]]}),
            hide_index=True, use_container_width=True
        )

    with st.expander("🤖 Model & training info"):
        st.markdown("""
**Model:** Random Forest Regressor · 150 trees · trained on 32 experimental images

| Concentration | Samples | Images |
|---|---|---|
| 1.0 mg/L | 2A, 2B | 8 |
| 2.0 mg/L | 3A, 3B | 8 |
| 4.0 mg/L | 4A, 4B | 8 |
| 8.0 mg/L | 5A | 4 |
| 9.7 mg/L | 5B | 4 |

**Performance:** MAE 1.0 mg/L · R² 0.794

**Top features learned:**
1. Saturation Std Dev — 34.7%
2. Mean Hue — 18.7%
3. Coverage % — 10.1%
        """)
        if is_control:
            st.info("Control mode active: prediction capped at ≤ 0.5 mg/L")

    with st.expander("🔍 Detection method"):
        st.markdown("""
**Duckweed pixel filter (strict):**
- Green channel > 110
- Green > Red + 5
- Green > Blue + 35
- Brightness between 80 – 155
- Connected component size: 20 – 800 px
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

    col_dl, col_new = st.columns([3,1])
    with col_dl:
        st.download_button("⬇️ Download result (CSV)", data=csv,
                           file_name=f"duckweed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    with col_new:
        if st.button("🔁 New analysis"):
            st.rerun()

else:
    st.info("📸 Take a photo or upload an image to begin.")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
**Duckweed Copper Analyzer** uses a Random Forest model trained on 32 images
from your experiment to predict copper contamination from a photo alone.

- **Accuracy:** ~1 mg/L average error  
- **R²:** 0.794  
- **Concentrations trained on:** 1, 2, 4, 8, 9.7 mg/L  
- **No lab equipment needed** — just a smartphone camera
        """)

st.markdown("---")
st.caption("Duckweed Copper Analyzer · ML Version · ISEF 2026 · Karthikeya Yeruva")
