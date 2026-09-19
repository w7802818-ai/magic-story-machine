# ISOM5240 Individual Assignment — Magic Story Machine

## Overview

**Magic Story Machine** is a Streamlit web application that generates children's stories from images using Hugging Face Transformers pipelines. Designed for children aged 3–10, the app:

1. Accepts an uploaded image
2. Generates a caption using the **Salesforce/blip-image-captioning-base** model
3. Expands the caption into a 50–100 word story using **GPT-2**
4. Converts the story to speech using **gTTS** (Google Text-to-Speech)

---

## Project Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme and server settings
└── README.md               # This file
```

---

## Local Setup

### Prerequisites

- Python 3.9 or later
- pip (Python package manager)

### Installation

1. Clone or download this project.

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

---

## Deploy to Streamlit Cloud

1. Push this project to a **public** GitHub repository.

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app** and fill in:

   | Field | Value |
   |-------|-------|
   | Repository | your-username/your-repo |
   | Branch | main |
   | Main file path | app.py |

4. Click **Deploy**. The first build will take a few minutes to download the models.

5. Once deployed, copy the URL and submit it with your assignment.

> **Note:** The models (BLIP ≈ 850 MB, GPT-2 ≈ 500 MB) are downloaded automatically on first run. Streamlit Cloud caches them between restarts.

---

## Models Used

| Task | Model | Source |
|------|-------|--------|
| Image Captioning | Salesforce/blip-image-captioning-base | [Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Story Generation | gpt2 | [Hugging Face](https://huggingface.co/gpt2) |
| Text-to-Speech | gTTS | [PyPI](https://pypi.org/project/gTTS/) |

---

## How It Works

1. **Image Upload** — The user uploads an image via the Streamlit file uploader.
2. **Image Captioning** — The BLIP model analyses the image and produces a one-sentence description.
3. **Story Generation** — The caption is fed into GPT-2 with a child-friendly prompt; the output is post-processed to stay within 50–100 words.
4. **Text-to-Speech** — gTTS converts the story into an MP3 file that can be played directly in the browser.

---

## Assessment Criteria Coverage

| Criterion | How it is addressed |
|-----------|-------------------|
| **Functionality** | Full pipeline: image → caption → story → audio |
| **Code Quality** | Modular functions, meaningful names, docstrings, comments |
| **Model Usage** | BLIP for captioning, GPT-2 for generation, gTTS for audio |
| **User Experience** | Animated gradient background, large buttons, card layout, emojis |
| **Deployment** | Ready for Streamlit Cloud with `requirements.txt` and `.streamlit/config.toml` |

---

## Submission Checklist

- [ ] `app.py` — source code with comments
- [ ] `requirements.txt` — all dependencies listed
- [ ] Streamlit Cloud URL — deployed and accessible
- [ ] All functionalities tested (upload → story → audio)

---

## Student Information

- **Name:** [Your Name]
- **Student ID:** [Your ID]
- **Email:** [Your Email]
- **Streamlit Cloud URL:** [Your URL]
