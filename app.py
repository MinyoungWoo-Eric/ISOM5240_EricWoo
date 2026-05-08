"""
Storytelling App  -  ISOM5240 Individual Assignment
====================================================
A Streamlit app that turns an uploaded picture into a short, kid-friendly
audio story for children aged 3 to 10.

Pipeline (each stage is a clearly named function):
    1) img2text(image)    : Salesforce/blip-image-captioning-base   (HF Image-to-Text)
    2) text2story(caption): Qwen/Qwen2.5-0.5B-Instruct              (HF Text Generation)
    3) text2audio(story)  : gTTS  (assignment-allowed, kid-friendly)

State management:
    - All HF models are loaded once via @st.cache_resource.
    - Caption / story / audio are kept in st.session_state so that:
        * Playing the audio does NOT regenerate the story.
        * Uploading a new image resets everything.
"""

# =========================================================================
# Imports
# =========================================================================
import hashlib
import io
import re

import streamlit as st
import torch
from gtts import gTTS
from PIL import Image
from transformers import pipeline


# =========================================================================
# Page configuration & light styling
# =========================================================================
st.set_page_config(
    page_title="Storytelling App",
    page_icon="📖",
    layout="centered",
)

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
def load_caption_model():
    """Load BLIP via the high-level pipeline helper."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


@st.cache_resource(show_spinner="✍️  Loading the story-teller…")
def load_story_pipeline():
    """
    Load the Qwen2.5-0.5B-Instruct text-generation pipeline.
    Qwen2.5 is a modern, instruction-tuned causal LM in the
    Hugging Face 'Text Generation' category.
    """
    # High-level pipeline helper (same as the HF model card snippet).
    # bfloat16 cuts the LM's RAM usage in half — important on
    # Streamlit Cloud's 1 GB free tier.
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
    )


# =========================================================================
# Small helpers
# =========================================================================
def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _trim_to_word_limit(text: str, max_words: int) -> str:
    """Trim text to <= max_words while preserving sentence boundaries."""
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
        words = text.split()[:max_words]
        trimmed = " ".join(words).rstrip(",;:") + "."
    return trimmed


def _hash_uploaded_file(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def _extract_assistant_reply(generated):
    """
    Pipeline output for chat-formatted input is a list of message dicts.
    Pull out the assistant's text content robustly.
    """
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict) and "content" in last:
            return last["content"]
    if isinstance(generated, str):
        return generated
    return str(generated)


# =========================================================================
# Stage 1 : Image  ->  Caption
# =========================================================================
def img2text(image) -> str:
    """Generate a short caption from a PIL.Image (or file path / bytes)."""
    captioner = load_caption_model()
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    result = captioner(image)
    return result[0]["generated_text"].strip()


# =========================================================================
# Stage 2 : Caption  ->  Kid-friendly 50-100 word story
# =========================================================================
def text2story(
    caption: str,
    min_words: int = 50,
    max_words: int = 100,
    max_attempts: int = 3,
) -> str:
    """
    Turn the image caption into a warm 50-100 word story for kids 3-10.
    Uses Qwen2.5-Instruct's chat template + retry loop + safety net.
    """
    storyteller = load_story_pipeline()

    system_prompt = (
        "You are a warm, friendly children's book author for kids aged 3 to 10. "
        "You write happy, simple, imaginative stories with cute character names "
        "and warm endings. You never include anything scary, sad, violent, or "
        "inappropriate. Your stories ALWAYS take place in the exact setting shown "
        "in the picture, and ALWAYS feature the same characters, animals, and "
        "objects mentioned in the picture description."
    )
    user_prompt = (
        f"Picture description: \"{caption}\"\n\n"
        f"Write a happy little story that is directly about this picture.\n"
        f"Requirements:\n"
        f"- The setting MUST match the picture description.\n"
        f"- The main characters MUST be those described in the picture.\n"
        f"- The story must be 70 to 90 words long.\n"
        f"- Use simple words and short sentences.\n"
        f"- Give the main character a cute name.\n"
        f"- End with a warm, happy ending.\n\n"
        f"Output only the story text — no title, no notes, no preamble."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_candidate = ""
    for attempt in range(max_attempts):
        outputs = storyteller(
            messages,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.70 + 0.05 * attempt,   
            top_p=0.95,
            repetition_penalty=1.15,
        )
        candidate = _extract_assistant_reply(outputs[0]["generated_text"]).strip()

        # Strip any leading "Story:" or quote marks the model may add.
        candidate = re.sub(r'^(story\s*:\s*|["“”\'])+', '', candidate, flags=re.I).strip()
        candidate = candidate.rstrip('"”\'')

        wc = _count_words(candidate)
        if wc > max_words:
            candidate = _trim_to_word_limit(candidate, max_words)
            wc = _count_words(candidate)

        if min_words <= wc <= max_words:
            return candidate

        last_candidate = candidate

    # ---- Safety net: force the result into the 50-100 range. ----
    story = last_candidate
    if _count_words(story) > max_words:
        story = _trim_to_word_limit(story, max_words)

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
# Stage 3 : Story  ->  Audio  (gTTS - allowed by the assignment PDF)
# =========================================================================
def text2audio(story_text: str) -> bytes:
    """
    Convert a story into MP3 audio bytes using gTTS.
    `tld='com'` gives a warm Enlish female voice that sounds
    storyteller-like for children. Switch to 'co.uk' (Uk) or 'com.au'
    (Australian) if you prefer a different accent.
    """
    tts = gTTS(
        text=story_text,
        lang="en",
        tld="com",
        slow=False,
    )
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()


# =========================================================================
# Streamlit UI
# =========================================================================
st.title("📖 Magical Storytelling App")
st.markdown("#### Upload a picture and I'll turn it into a fun little story 🦄✨")
st.markdown(
    "Made for kids aged **3 to 10** — pick a theme, get a story, and listen!"
)

with st.sidebar:
    st.header("🎨 How it works")
    st.caption(
        "ℹ️  The first time you use the app, the AI models need to load. "
        "After that, everything is fast! ☕"
    )

uploaded_file = st.file_uploader(
    "📷 Choose a picture (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
)

for key in ("file_key", "caption", "story", "audio"):
    st.session_state.setdefault(key, None)


if uploaded_file is not None:
    file_key = _hash_uploaded_file(uploaded_file)

    # New image  ->  reset all generated content.
    if st.session_state["file_key"] != file_key:
        st.session_state.update(
            file_key=file_key,
            caption=None,
            story=None,
            audio=None,
        )

    # Always show the uploaded image (Requirement #4).
    st.image(uploaded_file, caption="🖼️  Uploaded Picture", use_container_width=True)

    # ---- Stage 1: caption (once per image) ----
    if st.session_state["caption"] is None:
        with st.spinner("🔍  Looking at your picture…"):
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.session_state["caption"] = img2text(pil_image)
            except Exception as exc:
                st.error(f"😔  Sorry, I couldn't read this picture. ({exc})")
                st.stop()
    st.success("✅  I see what's in your picture!")
    st.markdown("**🔎  What I saw in the picture:**")
    st.info(st.session_state["caption"])

    # ---- Stage 2: story (once per image+theme; or on user request) ----
    st.markdown("### 📚 Your story")

    regen_clicked = st.button(
        "🔄  Try a different story",
        use_container_width=True,
        help="Generate a brand-new story from the same picture",
    )

    needs_new_story = (
        st.session_state["story"] is None
        or regen_clicked
    )

    if needs_new_story:
        with st.spinner("📝  Writing your story…"):
            try:
                st.session_state["story"] = text2story(
                    st.session_state["caption"]
                )
                # Story changed -> any cached audio is now stale.
                st.session_state["audio"] = None
            except Exception as exc:
                st.error(f"😔  Sorry, I couldn't write a story. ({exc})")
                st.stop()

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

        # ---- Stage 3: audio (once per story; replays do NOT regenerate) ----
        st.markdown("### 🎧 Listen to your story")

        if st.session_state["audio"] is None:
            if st.button(
                "🔊  Read it aloud!",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("🎙️  Recording the story with a friendly voice…"):
                    try:
                        st.session_state["audio"] = text2audio(
                            st.session_state["story"]
                        )
                    except Exception as exc:
                        st.error(
                            f"😔  Sorry, I couldn't read the story aloud. "
                            f"Please check your internet connection. ({exc})"
                        )

        if st.session_state["audio"] is not None:
            st.audio(st.session_state["audio"], format="audio/mp3")
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
