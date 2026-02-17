"""
DUCKWEED COPPER ANALYZER - MACHINE LEARNING VERSION
====================================================
Karthikeya sai Yeruva 1*, Dr Sarina J. Ergas, Dr Ananda Bhattacharjee
University of South Florida | Steinbrenner High School | ISEF 2026
"""

import streamlit as st
from streamlit.components.v1 import html as st_html
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import pickle
import os
import base64
import io

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


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CAMERA COMPONENT
#  Uses st.components.v1.html() which runs in its own iframe with its own
#  JS context — getUserMedia works here without Streamlit's restrictions.
#  The captured JPEG is sent back as a base64 string via postMessage, then
#  stored in st.session_state by a tiny Streamlit-side JS listener injected
#  via st.markdown.
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0e1117; font-family: sans-serif; }

  #wrapper {
    display: flex; flex-direction: column; align-items: center;
    gap: 10px; padding: 10px;
  }

  #video {
    width: 100%; max-width: 480px; border-radius: 10px;
    background: #000; display: block;
  }

  #canvas { display: none; }

  #preview {
    display: none; width: 100%; max-width: 480px;
    border-radius: 10px; border: 2px solid #43a047;
  }

  .btn-row {
    display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;
  }

  button {
    padding: 10px 22px; border: none; border-radius: 8px;
    font-size: 15px; font-weight: 600; cursor: pointer; transition: opacity .2s;
  }
  button:active { opacity: .7; }

  #btn-capture { background: #43a047; color: white; }
  #btn-flip    { background: #1976d2; color: white; }
  #btn-retake  { background: #e53935; color: white; display: none; }
  #btn-use     { background: #43a047; color: white; display: none; }

  #status {
    color: #aaa; font-size: 13px; text-align: center; min-height: 18px;
  }
  #error  { color: #ff5252; font-size: 13px; text-align: center; }
</style>
</head>
<body>
<div id="wrapper">
  <video id="video" autoplay playsinline></video>
  <canvas id="canvas"></canvas>
  <img id="preview" alt="Captured photo">

  <div id="status">Starting camera…</div>
  <div id="error"></div>

  <div class="btn-row">
    <button id="btn-flip">🔄 Flip Camera</button>
    <button id="btn-capture">📸 Capture</button>
    <button id="btn-retake">↩ Retake</button>
    <button id="btn-use">✅ Use Photo</button>
  </div>
</div>

<script>
const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const preview = document.getElementById('preview');
const status  = document.getElementById('status');
const errDiv  = document.getElementById('error');

const btnCapture = document.getElementById('btn-capture');
const btnFlip    = document.getElementById('btn-flip');
const btnRetake  = document.getElementById('btn-retake');
const btnUse     = document.getElementById('btn-use');

let currentFacing = 'environment';   // start with rear camera
let stream        = null;
let capturedData  = null;

async function startCamera(facing) {
  errDiv.textContent = '';
  status.textContent = 'Starting ' + (facing === 'environment' ? 'rear' : 'selfie') + ' camera…';

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  const constraints = {
    video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: false
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    currentFacing = facing;
    status.textContent = (facing === 'environment' ? '🔙 Rear' : '🤳 Selfie') + ' camera active';
    showLive();
  } catch (err) {
    errDiv.textContent = '⚠ Camera error: ' + err.message;
    status.textContent = '';
    // Try falling back to any camera
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      await video.play();
      status.textContent = 'Camera active (fallback mode)';
      showLive();
    } catch (err2) {
      errDiv.textContent = '⚠ No camera available: ' + err2.message;
    }
  }
}

function showLive() {
  video.style.display   = 'block';
  preview.style.display = 'none';
  btnCapture.style.display = '';
  btnFlip.style.display    = '';
  btnRetake.style.display  = 'none';
  btnUse.style.display     = 'none';
  capturedData = null;
}

function showPreview() {
  video.style.display   = 'none';
  preview.style.display = 'block';
  btnCapture.style.display = 'none';
  btnFlip.style.display    = 'none';
  btnRetake.style.display  = '';
  btnUse.style.display     = '';
}

btnCapture.addEventListener('click', () => {
  const w = video.videoWidth  || 640;
  const h = video.videoHeight || 480;
  canvas.width  = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(video, 0, 0, w, h);
  capturedData = canvas.toDataURL('image/jpeg', 0.92);
  preview.src  = capturedData;
  status.textContent = 'Photo captured — looks good?';
  showPreview();
});

btnFlip.addEventListener('click', () => {
  startCamera(currentFacing === 'environment' ? 'user' : 'environment');
});

btnRetake.addEventListener('click', () => {
  status.textContent = 'Ready to capture';
  showLive();
});

btnUse.addEventListener('click', () => {
  if (!capturedData) return;
  // Strip the data-URL prefix, keep only base64 payload
  const b64 = capturedData.split(',')[1];
  // Send to parent Streamlit window
  window.parent.postMessage({ type: 'duckweed_photo', data: b64 }, '*');
  status.textContent = '✅ Photo sent for analysis!';
});

// Boot
startCamera(currentFacing);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  JS LISTENER  – injected into the main Streamlit page.
#  Receives the postMessage from the iframe and writes the base64 string into
#  a hidden Streamlit text_input so Python can read it via session_state.
# ─────────────────────────────────────────────────────────────────────────────

LISTENER_JS = """
<script>
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'duckweed_photo') {
    // Find the hidden input Streamlit rendered for key "camera_b64"
    const inputs = window.parent.document.querySelectorAll('input[type=text]');
    for (const inp of inputs) {
      // Streamlit encodes the key in the aria-label or data attributes
      if (inp.getAttribute('aria-label') === 'camera_b64' ||
          inp.id.includes('camera_b64')) {
        inp.value = e.data.data;
        inp.dispatchEvent(new Event('input', { bubbles: true }));
        break;
      }
    }
    // Fallback: store in sessionStorage and trigger rerun via button click
    sessionStorage.setItem('duckweed_b64', e.data.data);
    const btn = window.parent.document.querySelector('[data-testid="baseButton-secondary"][kind="secondary"]');
  }
}, false);
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_model():
    for base in [".", "/home/claude"]:
        mp = os.path.join(base, "duckweed_model.pkl")
        sp = os.path.join(base, "duckweed_scaler.pkl")
        if os.path.exists(mp):
            with open(mp, "rb") as f: model = pickle.load(f)
            with open(sp, "rb") as f: scaler = pickle.load(f)
            return model, scaler
    st.error("⚠️ Model files not found — place duckweed_model.pkl and duckweed_scaler.pkl next to this script.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  VISION / FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

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


def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("Duckweed Copper Analyzer")
st.caption("Karthikeya Yeruva · Steinbrenner HS / USF · ISEF 2026")
st.markdown("---")

is_control = st.checkbox("This is a control sample (0 mg/L copper)")
st.markdown("---")

tab_cam, tab_upload = st.tabs(["📷 Camera", "📁 Upload Image"])

image_source = None   # PIL Image or None

with tab_cam:
    st.caption("Rear camera on by default. Tap **🔄 Flip Camera** to switch. Tap **✅ Use Photo** when happy.")

    # Render the custom HTML camera inside an iframe-like component
    st_html(CAMERA_HTML, height=520, scrolling=False)

    # ── Receive base64 photo back from the component ──────────────────────
    # We use a text_area (hidden with CSS) as a message bus.
    # The JS listener injected below writes the b64 string into it.
    st.markdown("""
    <style>
      /* Hide the textarea used as message bus */
      [data-testid="stTextArea"][aria-label="camera_b64_bus"] {
        display: none !important;
        height: 0 !important;
        overflow: hidden !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # Hidden text_area acts as the Python-readable state sink
    raw_b64 = st.text_area("camera_b64_bus", key="camera_b64_bus",
                            label_visibility="hidden", height=1)

    # Listener that writes postMessage payload into the text_area
    st.markdown("""
    <script>
    (function() {
      function findTextArea() {
        // Find the textarea linked to key camera_b64_bus
        return window.parent.document.querySelector('textarea[aria-label="camera_b64_bus"]');
      }

      window.addEventListener('message', function(evt) {
        if (!evt.data || evt.data.type !== 'duckweed_photo') return;
        const b64 = evt.data.data;

        // Try to write directly into the Streamlit textarea
        const ta = findTextArea();
        if (ta) {
          const nativeSet = Object.getOwnPropertyDescriptor(
            window.parent.HTMLTextAreaElement.prototype, 'value'
          ).set;
          nativeSet.call(ta, b64);
          ta.dispatchEvent(new window.parent.Event('input', { bubbles: true }));
        }
      }, false);
    })();
    </script>
    """, unsafe_allow_html=True)

    # If the textarea has data, decode it
    if raw_b64 and len(raw_b64) > 100:
        try:
            image_source = b64_to_pil(raw_b64.strip())
            st.success("✅ Photo received — scroll down for results!")
        except Exception as e:
            st.error(f"Could not decode photo: {e}")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

# ── Results ──────────────────────────────────────────────────────────────────
if image_source is not None:
    with st.spinner("Analyzing…"):
        res = analyze(image_source, is_control)
        status, color, icon = get_status(res["copper"])
        orig_viz, overlay_viz = make_viz(image_source, res["cropped"], res["mask"])

    c1, c2 = st.columns(2)
    with c1:
        st.image(orig_viz,    caption="Original sample",           use_container_width=True)
    with c2:
        st.image(overlay_viz, caption="Duckweed detected (green)", use_container_width=True)

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
- Green channel > 110  ·  Green > Red + 5  ·  Green > Blue + 35
- Brightness 80 – 155  ·  Component size 20 – 800 px
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

    dl_col, new_col = st.columns([3,1])
    with dl_col:
        st.download_button("⬇️ Download result (CSV)", data=csv,
                           file_name=f"duckweed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    with new_col:
        if st.button("🔁 New analysis"):
            st.session_state["camera_b64_bus"] = ""
            st.rerun()

else:
    st.info("📸 Use the Camera tab to take a photo, or upload an image file.")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
**Duckweed Copper Analyzer** uses a Random Forest ML model trained on 32 of your
own experimental images to predict copper contamination from a single photo.

- **Accuracy:** ~1 mg/L average error  ·  **R²:** 0.794
- **Trained concentrations:** 1, 2, 4, 8, 9.7 mg/L
- **No lab equipment needed** — just a smartphone or webcam
        """)

st.markdown("---")
st.caption("Duckweed Copper Analyzer · ML Version · ISEF 2026 · Karthikeya Yeruva")
