"""
Storytelling App  -  ISOM5240 Individual Assignment
====================================================
A Streamlit app that turns an uploaded picture into a short, kid-friendly
audio story for children aged 3 to 10.

Pipeline (each stage is a clearly named function):
    1) img2text(image)    : Salesforce/blip-image-captioning-base
    2) text2story(caption): google/flan-t5-base   (with prompt engineering
                            + 50-100 word validation/retry)
    3) text2audio(story)  : microsoft/speecht5_tts + a warm female speaker
                            embedding from the CMU-Arctic xvector dataset

State management:
    - All models are loaded once via @st.cache_resource.
    - Caption / story / audio are kept in st.session_state so that:
        * Playing the audio does NOT regenerate the story.
        * Changing the theme regenerates ONLY the story (not the caption).
        * Uploading a new image resets everything.
"""

# =========================================================================
# Imports
# =========================================================================
import hashlib
import re
from typing import Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from datasets import load_dataset
from transformers import pipeline


# =========================================================================
# Page configuration & light styling
# =========================================================================
st.set_page_config(
    page_title="Storytelling App",
    page_icon="📖",
    layout="centered",
)

# A small, kid-friendly stylesheet (rounded buttons, soft colors).
st.markdown(
    """
    <style>
        .main h1 { color: #6a4cff; }
        div.stButton > button {
            border-radius: 14px;
            padding: 0.6rem 1rem;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .story-card {
            background: linear-gradient(135deg, #FFF8E7 0%, #FFEAD5 100%);
            padding: 1.1rem 1.3rem;
            border-radius: 16px;
            border: 2px dashed #f5b971;
            font-size: 1.1rem;
            line-height: 1.65;
            color: #3a2e1f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Cached model loaders  (run once per session thanks to @st.cache_resource)
# =========================================================================
@st.cache_resource(show_spinner="🖼️  Loading the picture-reader…")
def load_caption_pipeline():
    """Load the image-to-text (BLIP) pipeline."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


@st.cache_resource(show_spinner="✍️  Loading the story-teller…")
def load_story_pipeline():
    """Load the instruction-tuned text2text (Flan-T5) pipeline."""
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
    )


@st.cache_resource(show_spinner="🎤  Loading the friendly voice…")
def load_tts_pipeline_and_embedding():
    """
    Load SpeechT5 TTS pipeline AND a warm female speaker embedding
    from the CMU-Arctic xvector dataset.
    Index 7306 is a bright, clear female voice that is much friendlier
    for kids than the original mms-tts-eng adult-male voice.
    """
    tts_pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embedding = torch.tensor(
        embeddings_dataset[7306]["xvector"]
    ).unsqueeze(0)
    return tts_pipe, speaker_embedding


# =========================================================================
# Small helpers
# =========================================================================
def _count_words(text: str) -> int:
    """Count words using a simple word-boundary regex."""
    return len(re.findall(r"\b\w+\b", text))


def _trim_to_word_limit(text: str, max_words: int) -> str:
    """
    Trim `text` to <= max_words while preserving sentence boundaries
    where possible. Falls back to a hard word cut if no full sentence fits.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chosen, count = [], 0
    for s in sentences:
        n = _count_words(s)
        if count + n > max_words:
            break
        chosen.append(s)
        count += n
    trimmed = " ".join(chosen).strip()
    if not trimmed:
        # No full sentence fits  ->  hard cut.
        words = text.split()[:max_words]
        trimmed = " ".join(words).rstrip(",;:") + "."
    return trimmed


def _hash_uploaded_file(uploaded_file) -> str:
    """Return an MD5 hash of the uploaded file (used as a state key)."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


# =========================================================================
# Stage 1 : Image  ->  Caption
# =========================================================================
def img2text(image) -> str:
    """
    Generate a short caption that describes the uploaded image.
    Accepts a PIL.Image, a numpy array, or a file path.
    """
    captioner = load_caption_pipeline()
    result = captioner(image)
    return result[0]["generated_text"].strip()


# =========================================================================
# Stage 2 : Caption  ->  Kid-friendly 50-100 word story
# =========================================================================
def text2story(
    caption: str,
    theme: str = "Adventure",
    min_words: int = 50,
    max_words: int = 100,
    max_attempts: int = 3,
) -> str:
    """
    Turn the image caption into a warm 50-100 word story for kids 3-10.

    Steps:
        1.  Build a kid-safe, theme-aware instruction prompt.
        2.  Generate with Flan-T5 (sampling for variety).
        3.  Validate word count; trim if too long; retry if too short.
        4.  Final safety net pads/trims so the result is always in range.
    """
    storyteller = load_story_pipeline()

    base_prompt = (
        f"Write a happy {theme.lower()} story for young children aged 3 to 10. "
        f"The story must be between 70 and 90 words. "
        f"It is about this picture: '{caption}'. "
        f"Use simple words and short sentences. "
        f"Give the main character a cute name. "
        f"Make it warm, imaginative, and friendly, with a happy ending. "
        f"Do not include anything scary, sad, or violent."
    )

    last_candidate = ""
    for attempt in range(max_attempts):
        outputs = storyteller(
            base_prompt,
            max_new_tokens=220,
            min_new_tokens=120,
            do_sample=True,
            temperature=0.85 + 0.05 * attempt,   # slightly more creative on retry
            top_p=0.95,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        candidate = outputs[0]["generated_text"].strip()

        # Some checkpoints echo the prompt — strip a leading "Story:" if present.
        if candidate.lower().startswith("story:"):
            candidate = candidate[len("story:"):].strip()

        wc = _count_words(candidate)
        if wc > max_words:
            candidate = _trim_to_word_limit(candidate, max_words)
            wc = _count_words(candidate)

        if min_words <= wc <= max_words:
            return candidate

        last_candidate = candidate  # keep the best we have so far

    # ---- Safety net: force the result into the 50-100 range. ----
    story = last_candidate
    if _count_words(story) > max_words:
        story = _trim_to_word_limit(story, max_words)

    # If still too short, append a gentle, safe closing line until we reach min.
    safe_endings = [
        "Everyone smiled and felt warm and happy inside.",
        "They held hands and laughed under the bright, kind sun.",
        "And from that day on, every day felt like a little gift.",
        "They waved goodnight, knowing tomorrow would be wonderful too.",
    ]
    i = 0
    while _count_words(story) < min_words and i < len(safe_endings):
        story = (story.rstrip(".!? ") + ". " + safe_endings[i]).strip()
        i += 1

    if _count_words(story) > max_words:
        story = _trim_to_word_limit(story, max_words)
    return story


# =========================================================================
# Stage 3 : Story  ->  Audio
# =========================================================================
def text2audio(story_text: str) -> Tuple[np.ndarray, int]:
    """
    Convert a story into audio (numpy array + sampling rate)
    using SpeechT5 with a warm female speaker embedding.
    """
    tts_pipe, speaker_embedding = load_tts_pipeline_and_embedding()
    output = tts_pipe(
        story_text,
        forward_params={"speaker_embeddings": speaker_embedding},
    )
    audio = np.asarray(output["audio"], dtype=np.float32)
    sampling_rate = int(output["sampling_rate"])
    return audio, sampling_rate


# =========================================================================
# Streamlit UI
# =========================================================================
st.title("📖 Magical Storytelling App")
st.markdown(
    "#### Upload a picture and I'll turn it into a fun little story 🦄✨"
)
st.markdown(
    "Made for kids aged **3 to 10** — pick a theme, get a story, and listen!"
)

# --- Sidebar : theme picker + info ---------------------------------------
with st.sidebar:
    st.header("🎨 Story settings")
    theme = st.selectbox(
        "Pick a story theme",
        ["Adventure", "Friendship", "Magic", "Animals", "Space", "Bedtime"],
        index=0,
    )
    st.markdown("---")
    st.caption(
        "ℹ️  The first time you use the app, the AI models need to load. "
        "After that, everything is fast! ☕"
    )

# --- File uploader -------------------------------------------------------
uploaded_file = st.file_uploader(
    "📷 Choose a picture (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
)

# --- Initialise session state slots --------------------------------------
for key in (
    "file_key", "caption", "story", "audio", "sample_rate", "theme_used",
):
    st.session_state.setdefault(key, None)


# =========================================================================
# Main flow (only runs when an image is uploaded)
# =========================================================================
if uploaded_file is not None:
    file_key = _hash_uploaded_file(uploaded_file)

    # If a different image was uploaded, reset all generated content.
    if st.session_state["file_key"] != file_key:
        st.session_state.update(
            file_key=file_key,
            caption=None,
            story=None,
            audio=None,
            sample_rate=None,
            theme_used=None,
        )

    # ---- Always show the uploaded image (Requirement #4) ----
    st.image(
        uploaded_file,
        caption="🖼️  Your picture",
        use_container_width=True,
    )

    # =====================================================================
    # Stage 1 : Caption (generated only once per image)
    # =====================================================================
    if st.session_state["caption"] is None:
        with st.spinner("🔍  Looking at your picture…"):
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.session_state["caption"] = img2text(pil_image)
            except Exception as exc:
                st.error(f"😔  Sorry, I couldn't read this picture. ({exc})")
                st.stop()
    st.success("✅  I see what's in your picture!")
    with st.expander("🔎  What I saw in the picture"):
        st.write(st.session_state["caption"])

    # =====================================================================
    # Stage 2 : Story (generated once per image+theme; or on user request)
    # =====================================================================
    st.markdown("### 📚 Your story")

    regen_clicked = st.button(
        "🔄  Try a different story",
        use_container_width=True,
        help="Generate a brand-new story from the same picture",
    )

    needs_new_story = (
        st.session_state["story"] is None
        or st.session_state["theme_used"] != theme
        or regen_clicked
    )

    if needs_new_story:
        with st.spinner("📝  Writing your story…"):
            try:
                st.session_state["story"] = text2story(
                    st.session_state["caption"], theme=theme
                )
                st.session_state["theme_used"] = theme
                # Story changed -> any cached audio is now stale.
                st.session_state["audio"] = None
                st.session_state["sample_rate"] = None
            except Exception as exc:
                st.error(f"😔  Sorry, I couldn't write a story. ({exc})")
                st.stop()

    # ---- Display the story + word count + download button ----
    if st.session_state["story"]:
        st.markdown(
            f"<div class='story-card'>{st.session_state['story']}</div>",
            unsafe_allow_html=True,
        )
        word_count = _count_words(st.session_state["story"])
        if 50 <= word_count <= 100:
            st.caption(f"📏  Word count: **{word_count}**  ✅ (target: 50–100)")
        else:
            st.caption(f"📏  Word count: **{word_count}**  (target: 50–100)")

        st.download_button(
            "💾  Download story as text",
            data=st.session_state["story"],
            file_name="my_story.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # =================================================================
        # Stage 3 : Audio (generated once per story; replay never regens)
        # =================================================================
        st.markdown("### 🎧 Listen to your story")

        # Show the "Read it aloud!" button only if no audio yet.
        if st.session_state["audio"] is None:
            if st.button(
                "🔊  Read it aloud!",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("🎙️  Recording the story with a friendly voice…"):
                    try:
                        audio, sr = text2audio(st.session_state["story"])
                        st.session_state["audio"] = audio
                        st.session_state["sample_rate"] = sr
                    except Exception as exc:
                        st.error(
                            f"😔  Sorry, I couldn't read the story aloud. ({exc})"
                        )

        # Once audio exists, just play it (replays do NOT regenerate anything).
        if st.session_state["audio"] is not None:
            st.audio(
                st.session_state["audio"],
                sample_rate=st.session_state["sample_rate"],
            )
            st.caption(
                "🎵  Press play above to listen — replay it as many times "
                "as you like!"
            )
else:
    st.info("👆  Please upload a picture to get started!")
    st.markdown(
        "**Try a picture of:** a puppy 🐶, a cat 🐱, a sunny beach 🏖️, "
        "your favourite toy 🧸, or a beautiful flower 🌸!"
    )
