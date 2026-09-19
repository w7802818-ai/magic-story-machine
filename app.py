"""
ISOM5240 Individual Assignment
Storytelling Application using Hugging Face Pipelines

This Streamlit application allows users to upload an image, generates a
children's story based on the image content using Hugging Face models,
and converts the story to speech for an engaging experience.

Designed for children aged 3-10 years old.

Models used:
    - Image Captioning: Salesforce/blip-image-captioning-base
    - Story Generation: gpt2
    - Text-to-Speech: gTTS (Google Text-to-Speech)

Author: [Student Name]
Date:   [Submission Date]
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
import os

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

/* ---- Story text ---- */
.story-text {
    font-size: 1.35rem;
    line-height: 2;
    color: #2d3436;
    text-align: left;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-radius: 16px;
    border-left: 5px solid #e17055;
    font-family: 'Georgia', serif;
}

/* ---- Caption text ---- */
.caption-text {
    font-size: 0.95rem;
    color: #636e72;
    text-align: center;
    font-style: italic;
    padding: 0.5rem;
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
if "uploaded"    not in st.session_state: st.session_state.uploaded    = False

# ============================================================
# Model Loading (cached — loaded once per session)
# ============================================================

@st.cache_resource(show_spinner="Loading image captioning model ...")
def load_captioning_model():
    """Load the BLIP image-captioning pipeline from Hugging Face."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


@st.cache_resource(show_spinner="Loading story generation model ...")
def load_story_model():
    """Load the GPT-2 text-generation pipeline from Hugging Face."""
    return pipeline(
        "text-generation",
        model="gpt2",
    )


def generate_audio_file(text, filename="story.mp3"):
    """
    Convert *text* to an MP3 audio file via gTTS.

    Returns the file path on success, or None on failure.
    """
    try:
        from gtts import gTTS

        tts  = gTTS(text=text, lang="en", slow=True)   # slow for kids
        path = os.path.join(os.getcwd(), filename)
        tts.save(path)
        return path
    except Exception as exc:
        st.error(f"Audio generation failed: {exc}")
        return None

# ============================================================
# Core Functions
# ============================================================

def clean_generated_text(text):
    """
    Remove common GPT-2 artefacts and truncate at a sentence boundary.
    """
    # Strip boiler-plate endings
    for marker in ["The end.", "The End.", "— The End",
                    "Once upon", "Once upon a"]:
        idx = text.rfind(marker)
        if idx > 0:
            text = text[:idx]

    # Cut at the last sentence-ending punctuation
    for punct in [". ", "! ", "? "]:
        pos = text.rfind(punct)
        if pos > 0:
            text = text[: pos + 1]
            break

    return text.strip()


def get_image_caption(image, caption_pipe):
    """
    Generate a descriptive caption for *image* using the BLIP model.

    Parameters
    ----------
    image : PIL.Image.Image
        The uploaded image.
    caption_pipe : transformers.Pipeline
        Pre-loaded image-to-text pipeline.

    Returns
    -------
    str  – a short caption describing the image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    result  = caption_pipe(image, max_new_tokens=50)
    caption = result[0]["generated_text"].strip()
    return caption


def generate_story(caption, story_pipe):
    """
    Expand *caption* into a short children's story (50-100 words).

    Parameters
    ----------
    caption : str
        Image caption produced by ``get_image_caption``.
    story_pipe : transformers.Pipeline
        Pre-loaded text-generation pipeline (GPT-2).

    Returns
    -------
    str  – a kid-friendly story.
    """
    prompt = (
        "Write a short, magical bedtime story for young children. "
        "Use simple, happy words. "
        "The story begins with: "
        f"{caption} "
        "Once upon a time, "
    )

    result = story_pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.85,
        top_p=0.92,
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
    )

    raw_text = result[0]["generated_text"]

    # Remove the prompt portion so only the new story remains
    story_body = raw_text[len(prompt):] if raw_text.startswith(prompt) else raw_text

    # Clean and post-process
    story_body = clean_generated_text(story_body)

    # ---- enforce 50-100 word window ----
    words = story_body.split()
    MAX_WORDS = 100

    if len(words) > MAX_WORDS:
        trimmed = " ".join(words[:MAX_WORDS])
        # Walk back to the last sentence boundary
        for punct in [".", "!", "?"]:
            pos = trimmed.rfind(punct)
            if pos > 0:
                story_body = trimmed[: pos + 1]
                break
        else:
            story_body = trimmed
    elif len(words) < 10 and caption:
        # Fallback: use caption as the story when generation is too short
        story_body = (
            f"Once upon a time, {caption} "
            "It was the most magical thing you could ever imagine! "
            "And they all lived happily ever after."
        )

    return story_body.strip()

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
        "Upload a picture and watch a magical story appear just for you!"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Upload Section ----
    st.markdown(
        '<div class="card">', unsafe_allow_html=True
    )
    st.markdown("### \U0001F4F7  Step 1 — Upload Your Picture!")

    uploaded_image = st.file_uploader(
        "Choose an image file (JPG, PNG, GIF, WebP)",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        help="Pick any photo — a pet, a landscape, your favourite toy …",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Image Preview ----
    if uploaded_image is not None:
        st.session_state.uploaded = True
        image = Image.open(uploaded_image)

        col_l, col_m, col_r = st.columns([1, 3, 1])
        with col_m:
            st.image(
                image,
                caption="Your beautiful picture!",
                use_container_width=True,
            )

        # ---- Action Button ----
        st.markdown("<br>", unsafe_allow_html=True)
        generate_col, _ = st.columns([1, 4])
        with generate_col:
            clicked = st.button(
                "\u2728  Create My Story!  \u2728",
                use_container_width=False,
            )

        if clicked:
            with st.spinner("\U0001F914  Looking at your picture …"):
                caption_pipe = load_captioning_model()
                caption = get_image_caption(image, caption_pipe)
                st.session_state.caption = caption

            with st.spinner("\u270D\uFE0F  Writing a magical story …"):
                story_pipe = load_story_model()
                story = generate_story(caption, story_pipe)
                st.session_state.story = story

            with st.spinner("\U0001F50A  Preparing the audio …"):
                audio_path = generate_audio_file(story)
                st.session_state.audio_file = audio_path

        # ---- Display Story ----
        if st.session_state.story:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="card">', unsafe_allow_html=True
            )
            st.markdown("### \U0001F4D6  Your Magical Story!")
            st.markdown(
                f'<div class="story-text">{st.session_state.story}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="caption-text">'
                f"\U0001F4DD Image description: {st.session_state.caption}"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Audio Player ----
            if st.session_state.audio_file:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="card">', unsafe_allow_html=True
                )
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
