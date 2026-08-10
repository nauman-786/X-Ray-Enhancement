import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --- Compatibility shim -----------------------------------------------------
# streamlit-drawable-canvas (last released 2023) calls
# streamlit.elements.image.image_to_url, which newer Streamlit versions
# removed/relocated. Patch it back in under its old name so the canvas
# library keeps working without downgrading Streamlit.
import streamlit.elements.image as _st_image_module
if not hasattr(_st_image_module, "image_to_url"):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
    except ImportError:
        from streamlit.elements.lib.image_utils import AtomicImage  # noqa: F401
        _image_to_url = None
    if _image_to_url is not None:
        _st_image_module.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="X-ray Enhancer", page_icon="🩻", layout="wide")


# ---------------------------------------------------------------------------
# Enhancement pipeline (unchanged)
# ---------------------------------------------------------------------------
def denoise(img, h=6):
    return cv2.fastNlMeansDenoising(img, h=h, templateWindowSize=7, searchWindowSize=21)


def build_pyramids_and_enhance(img, levels=4, gains=None):
    if gains is None:
        gains = [1.2, 1.8, 2.0, 1.5, 1.0]

    pyr = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img.astype(np.float32))

    lap_pyr = []
    for i in range(levels):
        size = (pyr[i].shape[1], pyr[i].shape[0])
        expanded = cv2.pyrUp(pyr[i + 1], dstsize=size)
        lap_pyr.append(pyr[i] - expanded)
    lap_pyr.append(pyr[-1])

    boosted = [lap * gains[i] if i < len(gains) else lap for i, lap in enumerate(lap_pyr)]

    out = boosted[-1]
    for i in range(len(boosted) - 2, -1, -1):
        size = (boosted[i].shape[1], boosted[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + boosted[i]

    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_xray(img, denoise_h=6, clahe_clip=2.0, sharpen_amount=1.5):
    stages = {"original": img}

    denoised = denoise(img, h=denoise_h)
    stages["denoised"] = denoised

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    local_contrast = clahe.apply(denoised)
    stages["local_contrast"] = local_contrast

    multi = build_pyramids_and_enhance(local_contrast)
    stages["multiscale"] = multi

    blurred = cv2.GaussianBlur(multi, (0, 0), 3)
    sharp = cv2.addWeighted(multi, sharpen_amount, blurred, -(sharpen_amount - 1.0), 0)
    stages["sharpened"] = sharp

    final = cv2.bilateralFilter(sharp, d=5, sigmaColor=30, sigmaSpace=30)
    stages["final"] = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return stages


# ---------------------------------------------------------------------------
# Radiologist viewing tools
# ---------------------------------------------------------------------------
def apply_window(img, level, width):
    """Classic PACS-style window/level contrast mapping."""
    width = max(width, 1)
    low = level - width / 2
    high = level + width / 2
    windowed = np.clip(img.astype(np.float32), low, high)
    windowed = (windowed - low) / (high - low) * 255.0
    return windowed.astype(np.uint8)


def roi_stats(img, x0, y0, x1, y1):
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    region = img[y0:y1, x0:x1]
    return {
        "width_px": x1 - x0,
        "height_px": y1 - y0,
        "mean": float(np.mean(region)),
        "std": float(np.std(region)),
        "min": int(np.min(region)),
        "max": int(np.max(region)),
        "median": float(np.median(region)),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🩻 X-ray Image Enhancer & Viewer")
st.caption(
    "Upload an X-ray to sharpen edges, boost local contrast, and inspect it with "
    "PACS-style viewing tools. For visibility purposes only — not a diagnostic tool."
)

with st.sidebar:
    st.header("Enhancement Settings")
    denoise_h = st.slider("Denoise strength", 0, 20, 6,
                           help="Higher = smoother, less grain, but can soften fine detail")
    clahe_clip = st.slider("Contrast boost (CLAHE clip limit)", 0.5, 5.0, 2.0, 0.1)
    sharpen_amount = st.slider("Sharpening strength", 1.0, 2.5, 1.5, 0.1)
    show_all_stages = st.checkbox("Show all pipeline stages", value=False)

uploaded_file = st.file_uploader(
    "Choose an X-ray image", type=["png", "jpg", "jpeg", "jpe", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Couldn't read that file as an image. Try a PNG or JPEG.")
    else:
        with st.spinner("Enhancing..."):
            stages = enhance_xray(
                img,
                denoise_h=denoise_h,
                clahe_clip=clahe_clip,
                sharpen_amount=sharpen_amount,
            )

        if show_all_stages:
            order = ["original", "denoised", "local_contrast", "multiscale", "sharpened", "final"]
            cols = st.columns(len(order))
            for col, key in zip(cols, order):
                col.image(stages[key], caption=key.replace("_", " ").title(),
                          use_column_width=True)
        else:
            col1, col2 = st.columns(2)
            col1.image(stages["original"], caption="Original", use_column_width=True)
            col2.image(stages["final"], caption="Enhanced", use_column_width=True)

        success, buf = cv2.imencode(".png", stages["final"])
        if success:
            st.download_button(
                label="⬇ Download Enhanced Image",
                data=buf.tobytes(),
                file_name=f"enhanced_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
            )

        st.divider()
        st.header("🔬 Radiologist View")
        st.caption(
            "Windowing, inversion, and region-of-interest analysis applied on top of the "
            "enhanced image — the way a PACS viewer lets you inspect a film."
        )

        base_img = stages["final"]
        img_mean = float(np.mean(base_img))
        img_std = float(np.std(base_img))

        view_col, tool_col = st.columns([2, 1])

        with tool_col:
            st.subheader("Window / Level")
            preset = st.selectbox(
                "Preset",
                ["Full range", "High contrast (narrow)", "Soft-tissue emphasis (wide)",
                 "Bone-like emphasis (high level)", "Custom"],
            )

            if preset == "Full range":
                default_level, default_width = 127, 255
            elif preset == "High contrast (narrow)":
                default_level, default_width = int(img_mean), max(int(img_std * 2), 10)
            elif preset == "Soft-tissue emphasis (wide)":
                default_level, default_width = int(img_mean * 0.8), 220
            elif preset == "Bone-like emphasis (high level)":
                default_level, default_width = int(min(img_mean + img_std, 245)), 90
            else:
                default_level, default_width = int(img_mean), int(img_std * 2) or 100

            level = st.slider("Level (center)", 0, 255, int(np.clip(default_level, 0, 255)))
            width = st.slider("Width (range)", 1, 510, int(np.clip(default_width, 1, 510)))
            invert = st.checkbox("Invert (negative)", value=False)

            windowed = apply_window(base_img, level, width)
            if invert:
                windowed = 255 - windowed

            st.download_button(
                "⬇ Download Windowed View",
                data=cv2.imencode(".png", windowed)[1].tobytes(),
                file_name=f"windowed_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
            )

        with view_col:
            st.subheader("Draw a region to inspect (ROI)")
            display_img = Image.fromarray(windowed)

            max_canvas_width = 650
            scale = min(1.0, max_canvas_width / display_img.width)
            canvas_w = int(display_img.width * scale)
            canvas_h = int(display_img.height * scale)

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.15)",
                stroke_width=2,
                stroke_color="#FFA500",
                background_image=display_img.resize((canvas_w, canvas_h)),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="rect",
                key="roi_canvas",
            )

        st.subheader("Selected Region")
        roi_found = False
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                obj = objects[-1]  # most recently drawn rectangle
                x0 = obj["left"] / scale
                y0 = obj["top"] / scale
                x1 = (obj["left"] + obj["width"] * obj.get("scaleX", 1)) / scale
                y1 = (obj["top"] + obj["height"] * obj.get("scaleY", 1)) / scale

                stats = roi_stats(windowed, x0, y0, x1, y1)
                if stats:
                    roi_found = True

                    # Crop the exact region and blow it up so it's clearly visible,
                    # regardless of how small the drawn box was.
                    xi0, xi1 = sorted((int(x0), int(x1)))
                    yi0, yi1 = sorted((int(y0), int(y1)))
                    xi0, yi0 = max(xi0, 0), max(yi0, 0)
                    xi1, yi1 = min(xi1, windowed.shape[1]), min(yi1, windowed.shape[0])
                    crop = windowed[yi0:yi1, xi0:xi1]

                    zoom_col, stats_col = st.columns([1, 1])

                    with zoom_col:
                        if crop.size > 0:
                            # Upscale small crops with nearest-neighbor-free interpolation
                            # so fine structure stays sharp, not blurry.
                            target_w = 500
                            zoom_factor = max(1, target_w // max(crop.shape[1], 1))
                            crop_zoomed = cv2.resize(
                                crop,
                                (crop.shape[1] * zoom_factor, crop.shape[0] * zoom_factor),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            st.image(crop_zoomed, caption="Zoomed ROI", use_column_width=True)
                            st.download_button(
                                "⬇ Download Zoomed ROI",
                                data=cv2.imencode(".png", crop_zoomed)[1].tobytes(),
                                file_name="roi_zoomed.png",
                                mime="image/png",
                                key="roi_download",
                            )

                    with stats_col:
                        st.metric("Width (px)", stats["width_px"])
                        st.metric("Height (px)", stats["height_px"])
                        m1, m2 = st.columns(2)
                        m1.metric("Mean", f"{stats['mean']:.1f}")
                        m2.metric("Median", f"{stats['median']:.1f}")
                        m3, m4 = st.columns(2)
                        m3.metric("Std Dev", f"{stats['std']:.1f}")
                        m4.metric("Min / Max", f"{stats['min']} / {stats['max']}")

        if not roi_found:
            st.info("Draw a rectangle on the image above to see a zoomed view and intensity stats for that region.")
else:
    st.info("Upload an image above to get started.")
