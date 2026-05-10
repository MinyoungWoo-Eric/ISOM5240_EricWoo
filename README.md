# 📖 Magical Storytelling App

Upload any picture and watch it turn into a fun little story — then listen to it being read aloud! 🦄✨

Built for kids aged **3 to 10** as part of the ISOM5240 Individual Assignment.

---

## ✨ What this app does

1. You **upload a picture** 🖼️
2. The app **looks at the picture** and writes down what it sees
3. It **writes a short story** (50–100 words) based on exactly what's in the picture
4. It **reads the story out loud** so you can sit back and listen 🎧

Three Hugging Face models power the three stages:

| Stage | Model | Type |
|---|---|---|
| 🖼️ Image captioning | `Salesforce/blip-image-captioning-base` | HF Image-to-Text |
| ✍️ Story generation | `Qwen/Qwen2.5-0.5B-Instruct` | HF Text Generation |
| 🎤 Text-to-speech | `gTTS` (Google Text-to-Speech) | Python TTS module |

---

## 🎨 Features

- 📷 Upload `.jpg`, `.jpeg`, or `.png` pictures
- 🖼️ The uploaded picture stays visible throughout the whole experience
- 🔎 See a plain-English description of what the AI spotted in your picture
- 📝 Get a warm, happy 50–100 word story based on your picture
- 📏 See the story's word count (the target is 50–100 words ✅)
- 🔄 Click **"Try a different story"** to get a brand-new story from the same picture
- 🔊 Click **"Read it aloud!"** to hear the story spoken in a friendly voice
- 🎵 Replay the audio as many times as you like — no waiting, no re-generating

---

## 🧒 For grown-ups: how to run it locally

### Requirements

- Python **3.10 or newer**
- About **1.5 GB of free disk space** (for the AI models, downloaded once)
- An internet connection (only needed for the initial model download and audio)

### Setup

```bash
# 1. Clone or download this repository
git clone <your-repo-url>
cd storytelling-app

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

> **First run is slow!** The AI models (~1 GB total) download on first use. After that, every run is fast thanks to Streamlit's `@st.cache_resource`. ☕

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push these files to a **public GitHub repository**:
   - `app.py`
   - `requirements.txt`
   - `README.md`

2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **"New app"**.

3. Select your repo, set the entry point to `app.py`, leave Python at the default (3.11+).

4. Click **Deploy**. The first build takes 5–10 minutes while models download.

> **Memory note:** Streamlit Community Cloud's free tier allows ~1 GB RAM. This app uses `blip-image-captioning-base` (~500 MB) and `Qwen2.5-0.5B-Instruct` in `bfloat16` (~500 MB), keeping the total within the limit.

---

## 📷 What pictures work best?

- Pets and animals 🐶🐱🐰
- Outdoor scenes — parks, beaches, gardens 🏖️🌳
- Toys, books, and everyday objects 🧸📚
- Simple, clear, colourful photos

> Blurry or very dark pictures may produce a vague caption, which leads to a more generic story. Clear, well-lit photos give the best results!

---

## 🛠️ How the code is structured

```
app.py
│
├── load_caption_model()   — loads BLIP once via @st.cache_resource
├── load_story_pipeline()  — loads Qwen2.5 once via @st.cache_resource
│
├── img2text(image)        — Stage 1: image → caption (BLIP)
├── text2story(caption)    — Stage 2: caption → 50-100 word story (Qwen2.5)
└── text2audio(story_text) — Stage 3: story → MP3 audio bytes (gTTS)
```

Key design decisions:

- **`@st.cache_resource`** — each model loads exactly once per session; no redundant downloads.
- **`st.session_state`** — caption, story, and audio are cached across reruns so that pressing play never re-generates the story, and clicking "Try a different story" only re-runs Stage 2.
- **Uploading a new image resets** all three outputs automatically (detected via an MD5 file hash).
- **Word-count validation** — the story goes through a retry loop and a safety-net trim/pad so it always lands in the 50–100 word target.
- **Error handling** — all three stages are wrapped in `try/except`; failures show a friendly message instead of a Python traceback.

---

## 📝 License

Provided for educational use as part of the **ISOM5240 Individual Assignment**.
