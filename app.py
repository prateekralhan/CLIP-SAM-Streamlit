import os
import cv2
import clip
import time
import torch
import warnings
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from segment_anything import build_sam, SamAutomaticMaskGenerator

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CLIP+SAM - WebApp",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto",
)

MODEL_CHECKPOINT = "model/sam_vit_h_4b8939.pth"
upload_path = "uploads/"


@st.cache_resource()
def mask_generate(MODEL_CHECKPOINT):
    # Download the model weights to load them here
    model_start_time = time.time()
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=MODEL_CHECKPOINT))
    model_end_time = time.time()
    print("-" * 50)
    print(
        f"Model downloaded successfully in {model_end_time - model_start_time} seconds."
    )
    return mask_generator


def generate_image_masks(image_path, mask_generator):
    img_mask_start_time = time.time()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    img_mask_end_time = time.time()
    print("-" * 50)
    print(
        f"Image mask generated successfully in {img_mask_end_time - img_mask_start_time} seconds."
    )
    return masks


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def load_CLIP():
    # Load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device.type} for inference...")
    clip_start_time = time.time()
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_end_time = time.time()
    print("-" * 50)
    print(f"CLIP loaded successfully in {clip_end_time - clip_start_time} seconds.")
    return model, preprocess, device


@torch.no_grad()
def retriev(elements: list[Image.Image], search_text: str) -> int:
    model, preprocess, device = load_CLIP()
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


mask_generator = mask_generate(MODEL_CHECKPOINT)

st.title("✨ CLIP + SAM 🏜")
st.info(" Let me help perform open vocabulary image segmentation. 😉")
col_a, col_b = st.columns(2)

prompt = st.text_input("Enter your text", "pear")
image_path = st.file_uploader("Upload Image 🚀", type=["png", "jpg", "bmp", "jpeg"])
if image_path is not None and (prompt is not None or len(prompt.strip()) != 0):
    with open(os.path.join(upload_path, image_path.name), "wb") as f:
        f.write((image_path).getbuffer())
    uploaded_image_path = os.path.abspath(os.path.join(upload_path, image_path.name))
    with st.spinner("Working... 💫"):
        # Cut out all masks
        image = Image.open(uploaded_image_path)
        cropped_boxes = []
        masks = generate_image_masks(uploaded_image_path, mask_generator)

        for mask in masks:
            cropped_boxes.append(
                segment_image(image, mask["segmentation"]).crop(
                    convert_box_xywh_to_xyxy(mask["bbox"])
                )
            )

        scores = retriev(cropped_boxes, str(prompt))
        indices = get_indices_of_values_above_threshold(scores, 0.05)

        segmentation_masks = []

        for seg_idx in indices:
            segmentation_mask_image = Image.fromarray(
                masks[seg_idx]["segmentation"].astype("uint8") * 255
            )
            segmentation_masks.append(segmentation_mask_image)

        original_image = Image.open(uploaded_image_path)
        overlay_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_color = (255, 0, 0, 200)

        draw = ImageDraw.Draw(overlay_image)
        for segmentation_mask_image in segmentation_masks:
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

        result_image = Image.alpha_composite(
            original_image.convert("RGBA"), overlay_image
        )
        np_image = np.array(result_image)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, width=500)
                st.success("Original Image")
            with col2:
                st.image(np_image, width=500)
                st.success("Output based on CLIP+SAM")
else:
    st.warning("⚠ Please upload your Image! 😯")


st.markdown(
    "<br><hr><center>Made with ❤️ by <a href='mailto:ralhanprateek@gmail.com?subject=CLIP+SAM WebApp!&body=Please specify the issue you are facing with the app.'><strong>Prateek Ralhan</strong></a> with the help of [segment-anything](https://github.com/facebookresearch/segment-anything/tree/main) built by [Meta Research](https://github.com/facebookresearch) and [CLIP](https://github.com/openai/CLIP) built by [OpenAI](https://github.com/openai) ✨</center><hr>",
    unsafe_allow_html=True,
)
