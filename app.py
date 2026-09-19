"""
ISOM5240 Individual Assignment
Storytelling Application using Hugging Face Models

This Streamlit application allows users to upload an image, generates a
description using Hugging Face BLIP model, then creates a children's story
(500+ words) based on the description using Mistral-7B-Instruct, and
converts the story to speech for an engaging experience.

Designed for children aged 3-10 years old.

Models used:
    - Image Captioning : Salesforce/blip-image-captioning-base (Hugging Face)
    - Story Generation : mistralai/Mistral-7B-Instruct-v0.3 (via HF Inference API)
    - Text-to-Speech   : gTTS (Google Text-to-Speech)

Author: [Student Name]
Date:   [Submission Date]
"""

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
import os
import tempfile

# ============================================================
# Page Configuration — must be the first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Magic Story Machine",
    page_icon="\U0001F4DA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Custom CSS — kid-friendly colourful theme
# ============================================================

CUSTOM_CSS = """
<style>
/* ---- Gradient background ---- */
.stApp {
    background: linear-gradient(135deg,
        #667eea 0%,  #764ba2 25%,
        #f093fb 50%, #4facfe 75%,
        #00f2fe 100%);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%;   }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%;   }
}

/* ---- Hide default header / footer ---- */
header[data-testid="stHeader"]  { background: rgba(0,0,0,0); }
footer { visibility: hidden; }

/* ---- Title styling ---- */
.main-title {
    text-align: center;
    color: white;
    font-size: 2.8rem;
    font-weight: 800;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    margin-bottom: 0;
}
.sub-title {
    text-align: center;
    color: #fff;
    font-size: 1.15rem;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    margin-bottom: 1.5rem;
}

/* ---- Card container ---- */
.card {
    background: rgba(255,255,255,0.93);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* ---- Description box (BLIP output) ---- */
.description-box {
    font-size: 1.15rem;
    line-height: 1.8;
    color: #2d3436;
    text-align: left;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    border-radius: 16px;
    border-left: 5px solid #00b894;
    font-family: 'Georgia', serif;
}

/* ---- Story text ---- */
.story-text {
    font-size: 1.25rem;
    line-height: 2;
    color: #2d3436;
    text-align: left;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-radius: 16px;
    border-left: 5px solid #e17055;
    font-family: 'Georgia', serif;
    white-space: pre-wrap;
}

/* ---- Word count badge ---- */
.word-count {
    text-align: right;
    font-size: 0.9rem;
    color: #636e72;
    margin-top: 0.3rem;
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 50px;
    padding: 0.65rem 2rem;
    font-size: 1.15rem;
    font-weight: 700;
    border: none;
    color: white;
    background: linear-gradient(135deg, #6c5ce7, #a29bfe);
    transition: all 0.3s ease;
    cursor: pointer;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(108,92,231,0.4);
}

/* ---- File uploader ---- */
.stFileUploader {
    background: rgba(255,255,255,0.85);
    border-radius: 16px;
    padding: 1rem;
}

/* ---- Markdown overrides ---- */
h1, h2, h3 { color: #2d3436 !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# Session State — persist data across reruns
# ============================================================
if "story"       not in st.session_state: st.session_state.story       = None
if "caption"     not in st.session_state: st.session_state.caption     = None
if "audio_file"  not in st.session_state: st.session_state.audio_file  = None

# ============================================================
# Model Loading (cached — loaded once per session)
# ============================================================

@st.cache_resource(show_spinner="Loading image captioning model ...")
def load_captioning_model():
    """
    Load the BLIP image-captioning processor and model from Hugging Face.

    Uses BlipProcessor + BlipForConditionalGeneration directly for maximum
    compatibility across transformers versions.
    """
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model     = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


@st.cache_resource(show_spinner="Connecting to story generation service ...")
def get_story_client():
    """
    Return a Hugging Face Inference API client using the free serverless
    text_generation endpoint — no API key or token required.

    The model runs on HF servers, so no local download or GPU is needed.
    """
    return InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
    )


# ============================================================
# Core Functions
# ============================================================

def get_image_caption(image, processor, model):
    """
    Generate a descriptive caption for *image* using the BLIP model.

    Uses unconditional captioning so the description accurately reflects
    the actual image content.  The result is displayed on the page for
    the user to see before the story is generated.

    Parameters
    ----------
    image : PIL.Image.Image
    processor : BlipProcessor
    model : BlipForConditionalGeneration

    Returns
    -------
    str  – a sentence describing the image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Unconditional captioning — most faithful to the image
    inputs = processor(image, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=3,
    )
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption


def generate_story_from_caption(caption, client):
    """
    Use Hugging Face Inference API (Mistral-7B) to generate a children's story
    (>= 500 words) that is directly based on the BLIP image description.

    Uses the free serverless text_generation endpoint — no API key required.

    Parameters
    ----------
    caption : str
        Image description produced by the BLIP model.
    client : huggingface_hub.InferenceClient
        HF Inference API client (already configured with model).

    Returns
    -------
    str  – a children's story of at least 500 words.
    """
    # Build the prompt in Mistral's chat template format
    prompt = (
        "<s>[INST] You are a beloved children's storyteller who writes vivid, "
        "magical stories for kids aged 3 to 10. Your stories are warm, "
        "fun, and full of wonder. Use simple vocabulary that young "
        "children can understand, but make the storytelling rich and "
        "engaging with sensory details, dialogue, and gentle humour.\n\n"
        f"An image was analysed by an AI and the following description was produced:\n\n"
        f"\"{caption}\"\n\n"
        f"Based EXACTLY on this description, write a complete children's story. "
        f"Requirements:\n"
        f"1. The story MUST directly relate to the description above — use the same "
        f"characters, setting, and key elements mentioned.\n"
        f"2. The story MUST be at least 500 words long. Be creative and detailed.\n"
        f"3. Give the characters names and a simple personality.\n"
        f"4. Include a clear beginning, middle, and happy ending.\n"
        f"5. Use short sentences and simple words suitable for young children.\n"
        f"6. Add some dialogue between characters to make the story lively.\n"
        f"7. Do NOT include any scary or inappropriate content.\n\n"
        f"Write the full story now: [/INST]"
    )

    # Use the free serverless text_generation endpoint (no token needed)
    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=3000,
        temperature=0.8,
        top_p=0.9,
        return_full_text=False,
    )

    # text_generation with return_full_text=False returns the generated text directly
    if isinstance(response, str):
        story = response
    else:
        # Some versions return a dict or TextGenerationOutput object
        story = response.generated_text if hasattr(response, "generated_text") else str(response)

    return story.strip()


def generate_audio_file(text):
    """
    Convert *text* to an MP3 audio file via gTTS.

    Uses a temporary file to ensure it works on Streamlit Cloud
    where the working directory may not be writable.

    Returns the file path on success, or None on failure.
    """
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang="en", slow=True)   # slow for kids
        path = os.path.join(tempfile.gettempdir(), "story.mp3")
        tts.save(path)
        return path
    except Exception as exc:
        st.error(f"Audio generation failed: {exc}")
        return None

# ============================================================
# Main Application — UI
# ============================================================

def main():
    """Render the Magic Story Machine interface."""

    # ---- Title ----
    st.markdown(
        '<p class="main-title">\U0001F328 Magic Story Machine \U0001F328</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">'
        "Upload a picture \u2192 AI describes it \u2192 A magical story appears!"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Upload Section ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### \U0001F4F7  Step 1 \u2014 Upload Your Picture!")

    uploaded_image = st.file_uploader(
        "Choose an image file (JPG, PNG, GIF, WebP)",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        help="Pick any photo \u2014 a pet, a landscape, your favourite toy \u2026",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Image Preview & Generation ----
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        col_l, col_m, col_r = st.columns([1, 3, 1])
        with col_m:
            st.image(image, caption="Your beautiful picture!", width="stretch")

        # ---- Action Button ----
        st.markdown("<br>", unsafe_allow_html=True)
        generate_col, _ = st.columns([1, 4])
        with generate_col:
            clicked = st.button("\u2728  Create My Story!  \u2728")

        if clicked:
            story_generated = False

            # --- Step 1: Image Captioning (BLIP) ---
            with st.spinner("\U0001F914  Looking at your picture \u2026"):
                processor, model = load_captioning_model()
                caption = get_image_caption(image, processor, model)
                st.session_state.caption = caption

            # --- Step 2: Story Generation (HF Inference API — Mistral-7B) ---
            with st.spinner("\u270D\uFE0F  Writing a magical story (this may take a moment) \u2026"):
                try:
                    client = get_story_client()
                    story = generate_story_from_caption(caption, client)
                    st.session_state.story = story
                    story_generated = True
                except Exception as exc:
                    st.error(f"Story generation failed: {exc}")
                    st.info("The Hugging Face Inference API may be busy. Please try again in a moment.")

            # --- Step 3: Text-to-Speech ---
            if story_generated:
                with st.spinner("\U0001F50A  Preparing the audio \u2026"):
                    audio_path = generate_audio_file(story)
                    st.session_state.audio_file = audio_path

        # ---- Display Image Description (BLIP output) ----
        if st.session_state.caption:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### \U0001F4DD  Step 2 \u2014 What the AI Sees")
            st.markdown(
                f'<div class="description-box">{st.session_state.caption}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="text-align:center; color:#636e72; font-size:0.9rem;">'
                "\u2B06 This description was generated from your image by "
                "Salesforce/blip-image-captioning-base"
                "</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- Display Story ----
        if st.session_state.story:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### \U0001F4D6  Step 3 \u2014 Your Magical Story!")

            st.markdown(
                f'<div class="story-text">{st.session_state.story}</div>',
                unsafe_allow_html=True,
            )

            # Word count
            word_count = len(st.session_state.story.split())
            st.markdown(
                f'<p class="word-count">Word count: {word_count}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="text-align:center; color:#636e72; font-size:0.9rem;">'
                "\u2B06 This story was generated by Mistral-7B (Hugging Face Inference API) based on the image description above"
                "</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Audio Player ----
            if st.session_state.audio_file:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### \U0001F3B5  Listen to Your Story!")
                with open(st.session_state.audio_file, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                st.markdown(
                    '<p style="text-align:center; font-size:1.1rem;">'
                    "\U0001F3A7 Press play and listen to your story!"
                    "</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Download Story Text ----
            st.download_button(
                label="\U0001F4E5  Download Story as Text File",
                data=st.session_state.story,
                file_name="my_magical_story.txt",
                mime="text/plain",
            )

    # ---- Footer ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:white; '
        "text-shadow: 1px 1px 2px rgba(0,0,0,0.3);\">"
        "\u2B50 Every picture has a story waiting to be told! \u2B50"
        "</p>",
        unsafe_allow_html=True,
    )


# Entry point
if __name__ == "__main__":
    main()
