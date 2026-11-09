import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import Predictor, open_image

BASE = Path(__file__).parent
WEIGHTS = BASE / "models" / "best_cnn_model.pth"
HISTORY = BASE / "models" / "training_history.json"

# Final numbers reported in the training notebook (test set, 2,495 images)
TEST_METRICS = {"accuracy": 0.9507, "macro_f1": 0.9507, "fpr": 0.0537}
DATA_SPLIT = {"Train": 19943, "Validation": 2492, "Test": 2495}

st.set_page_config(page_title="Pet Classifier | Cat vs Dog", page_icon="🐾", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 60%, #a21caf 100%);
       border-radius: 20px; padding: 2.2rem 2.5rem; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2.2rem; color: #fff;}
.hero p {margin: .4rem 0 0 0; color: #e9d5ff; font-size: 1.05rem;}
.card {background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 1.25rem 1.5rem;}
.metric {background: #111827; border: 1px solid #1f2937; border-radius: 16px;
         padding: 1rem 1.25rem; text-align: left;}
.metric .label {color: #9ca3af; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;}
.metric .value {color: #f9fafb; font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.metric .sub {color: #6b7280; font-size: .8rem;}
.pred {font-size: 2.4rem; font-weight: 800; margin: 0; color: #f9fafb;}
.badge {display: inline-block; padding: .2rem .75rem; border-radius: 999px;
        font-size: .8rem; font-weight: 600; margin-bottom: .6rem;}
.badge.High {background: #064e3b; color: #6ee7b7;}
.badge.Moderate {background: #78350f; color: #fcd34d;}
.badge.Low {background: #7f1d1d; color: #fca5a5;}
.bar-row {display: flex; justify-content: space-between; color: #d1d5db;
          font-size: .9rem; margin: .8rem 0 .25rem 0;}
.bar-track {background: #1f2937; border-radius: 999px; height: 10px; overflow: hidden;}
.bar-fill {height: 10px; border-radius: 999px;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding: .6rem 1.1rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> Predictor:
    return Predictor(WEIGHTS)


@st.cache_data
def load_history():
    if HISTORY.exists():
        return pd.DataFrame(json.loads(HISTORY.read_text()))
    return None


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def prob_bar(name: str, p: float, color: str) -> str:
    return (
        f'<div class="bar-row"><span>{name}</span><span>{p * 100:.1f}%</span></div>'
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{p * 100:.1f}%; background:{color};"></div></div>'
    )


def result_card(res):
    st.markdown(
        f'<div class="card"><span class="badge {res.level}">{res.level} confidence</span>'
        f'<p class="pred">{res.emoji} {res.label}</p>'
        f'<div style="color:#9ca3af;">Confidence {res.confidence * 100:.1f}%</div>'
        + prob_bar("🐱 Cat", res.p_cat, "#38bdf8")
        + prob_bar("🐶 Dog", res.p_dog, "#a78bfa")
        + "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ model
if not WEIGHTS.exists():
    st.error(f"Model weights not found at `{WEIGHTS}`. Put `best_cnn_model.pth` in the `models/` folder.")
    st.stop()

predictor = load_predictor()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🐾 Pet Classifier")
    st.caption("CNN trained from scratch with PyTorch")
    st.divider()
    st.markdown("### Settings")
    use_tta = st.toggle("Test-time augmentation", value=False,
                        help="Averages the prediction of the image and its mirror image. Slightly slower.")
    use_cam = st.toggle("Explain with Grad-CAM", value=True,
                        help="Shows which parts of the image influenced the decision.")
    st.divider()
    st.markdown("### Model")
    st.markdown(f"- Device: `{predictor.device}`")
    st.markdown(f"- Parameters: `{predictor.n_params / 1e6:.1f}M`")
    st.markdown("- Input: `224 x 224` RGB")
    st.divider()
    st.caption("Tip: use a clear, well-lit photo with one animal.")

# ------------------------------------------------------------------ hero
st.markdown(
    """
<div class="hero">
  <h1>🐱 Cat vs Dog 🐶 Image Classifier</h1>
  <p>Upload a photo and the model tells you what it sees, how sure it is, and where it looked.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔍 Classify", "📂 Batch", "📊 Model insights", "ℹ️ About"]
)

# ------------------------------------------------------------------ single
with tab_single:
    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
    if source == "Upload":
        file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Take a photo")

    if file is None:
        st.info("Add an image to get a prediction.")
    else:
        try:
            image = open_image(file)
        except Exception:
            st.error("This file could not be read as an image. Try a JPG or PNG.")
            st.stop()

        with st.spinner("Analyzing..."):
            res = predictor.predict(image, tta=use_tta)

        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.image(image, caption=f"Uploaded image ({image.width} x {image.height})", width="stretch")
        with right:
            result_card(res)
            if res.level == "Low":
                st.warning("Low confidence. The image may be unclear or may not show a cat or a dog. "
                           "The model only knows these two classes.")

        if use_cam:
            with st.expander("🔥 Why this prediction? (Grad-CAM)", expanded=True):
                try:
                    heat = predictor.gradcam(image)
                    c1, c2 = st.columns(2)
                    c1.image(image, caption="Original", width="stretch")
                    c2.image(heat, caption="Model attention (brighter = more important)",
                             width="stretch")
                except Exception as e:
                    st.caption(f"Grad-CAM is not available for this image ({type(e).__name__}).")

# ------------------------------------------------------------------ batch
with tab_batch:
    files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key="batch")
    if not files:
        st.info("Upload multiple images to classify them all at once.")
    else:
        rows, images = [], []
        progress = st.progress(0.0, text="Classifying...")
        for i, f in enumerate(files, start=1):
            try:
                img = open_image(f)
                r = predictor.predict(img, tta=use_tta)
                rows.append({"File": f.name, "Prediction": r.label,
                             "Confidence (%)": round(r.confidence * 100, 1),
                             "P(dog) (%)": round(r.p_dog * 100, 1)})
                images.append((img, r))
            except Exception:
                rows.append({"File": f.name, "Prediction": "Error",
                             "Confidence (%)": None, "P(dog) (%)": None})
            progress.progress(i / len(files), text=f"Classifying {i}/{len(files)}")
        progress.empty()

        df = pd.DataFrame(rows)
        ok = df[df["Prediction"] != "Error"]
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Images", str(len(df)))
        with m2: metric_card("Cats", str((ok["Prediction"] == "Cat").sum()), "🐱")
        with m3: metric_card("Dogs", str((ok["Prediction"] == "Dog").sum()), "🐶")
        with m4: metric_card("Avg confidence", f"{ok['Confidence (%)'].mean():.1f}%" if len(ok) else "-")

        st.markdown("&nbsp;")
        cols = st.columns(4)
        for i, (img, r) in enumerate(images):
            with cols[i % 4]:
                st.image(img, caption=f"{r.emoji} {r.label} · {r.confidence * 100:.0f}%",
                         width="stretch")

        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

# ------------------------------------------------------------------ insights
with tab_insights:
    st.markdown("### Test-set performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Accuracy", f"{TEST_METRICS['accuracy'] * 100:.2f}%", "2,495 test images")
    with c2: metric_card("Macro F1", f"{TEST_METRICS['macro_f1']:.4f}", "main target metric")
    with c3: metric_card("False positive rate", f"{TEST_METRICS['fpr'] * 100:.2f}%", "cats predicted as dogs")
    with c4: metric_card("Parameters", f"{predictor.n_params / 1e6:.1f}M", "trained from scratch")

    st.markdown("&nbsp;")
    hist = load_history()
    if hist is not None:
        st.markdown("### Training history")
        h1, h2 = st.columns(2)
        hist.index = range(1, len(hist) + 1)
        hist.index.name = "epoch"
        with h1:
            st.markdown("**Loss**")
            st.line_chart(hist[["loss", "val_loss"]].rename(columns={"loss": "train", "val_loss": "validation"}))
        with h2:
            st.markdown("**Macro F1**")
            st.line_chart(hist[["f1", "val_f1"]].rename(columns={"f1": "train", "val_f1": "validation"}))
        st.caption(f"Trained for {len(hist)} epochs. Best validation macro F1: {hist['val_f1'].max():.4f}.")
    else:
        st.caption("Add `models/training_history.json` to show the training curves.")

    st.markdown("### Data and architecture")
    d1, d2 = st.columns(2)
    with d1:
        st.dataframe(pd.DataFrame({"Split": list(DATA_SPLIT), "Images": list(DATA_SPLIT.values())}),
                     hide_index=True, width="stretch")
    with d2:
        st.code(
            "4 x [Conv3x3 (padding=1) -> BatchNorm -> ReLU -> MaxPool2]\n"
            "channels: 3 -> 32 -> 64 -> 128 -> 256\n"
            "Flatten (256x14x14) -> Linear 256 -> ReLU -> Dropout 0.5 -> Linear 1\n"
            "Loss: BCEWithLogits | Adam | ReduceLROnPlateau | Early stopping",
            language="text",
        )

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        """
### How it works
1. The image is converted to RGB, resized to **224 x 224** and normalized.
2. The CNN outputs one score. A sigmoid turns it into the probability of **dog**.
3. Above 50% the image is a dog, otherwise a cat. The confidence is the larger of the two probabilities.
4. **Grad-CAM** highlights the regions that pushed the model towards its answer.

### Good to know
- The model only knows **cats and dogs**. A photo of anything else is still forced into one of the two classes, so watch for low confidence.
- It was trained on photos with one clear animal. Cartoons, heavy filters and tiny animals in a large scene can confuse it.
- Test accuracy is about **95%**, so roughly 1 image in 20 can be wrong.
"""
    )
