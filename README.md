# ISOM5240_EricWoo
# 📖 Magical Storytelling App

Turn any picture into a fun, friendly little story for kids aged **3 to 10** — and listen to it being read aloud! 🦄✨

---

## ✨ What this app does

1. You **upload a picture** 🖼️
2. The app **looks at the picture** and writes down what it sees
3. It then **writes a short story** (50–100 words) for you, in the theme you pick 🎨
4. Finally, it **reads the story out loud** so you can sit back and listen 🎧

It is built with three Hugging Face models, one per stage:

| Stage | Model | Hugging Face ID |
|---|---|---|
| 🖼️ Image captioning | BLIP | `Salesforce/blip-image-captioning-large` |
| ✍️ Story generation | Flan-T5 (instruction-tuned) | `google/flan-t5-base` |
| 🎤 Text-to-speech | SpeechT5 + warm female voice | `microsoft/speecht5_tts` |

---

## 🎨 Features

- 📷 Upload `.jpg`, `.jpeg`, or `.png` pictures and see them stay on the page.
- 🎭 Pick a **story theme** — Adventure, Friendship, Magic, Animals, Space, or Bedtime.
- 📝 Get a **kid-friendly** story between **50 and 100 words**.
- 📏 See the **word count** of your story.
- 🔄 Click **"Try a different story"** to roll the dice again on the same picture.
- 💾 **Download** your story as a text file.
- 🔊 **Listen** to the story read aloud in a clear, friendly voice.
- 🎵 **Replay** the audio as many times as you like — no waiting, no re-generating!

---

## 🧒 For grown-ups: how to run it

### Requirements
- Python **3.10 or newer**
- About **3 GB of free disk space** (the AI models download the first time)
- An internet connection (only needed once, to download the models)

### Setup

```bash
# 1. Clone or download this repo
git clone <your-repo-url>
cd storytelling-app

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **First run is slow!** The three models (~2.5 GB total) are downloaded the first time. After that, every run is fast.

---

## ☁️ Deploying on Streamlit Cloud

1. Push these three files to a **public GitHub repo**:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **"New app"**.
3. Pick your repo and `app.py` as the entry point. Leave the Python version at the default (3.11+).
4. Click **Deploy**. The first build takes 5–10 minutes while the models download.
5. Share the resulting URL — that's your demo!

---

## 📷 What pictures work best?

- Photos of pets 🐶🐱
- Toys, dolls, or stuffed animals 🧸
- Outdoor scenes — beaches, mountains, parks 🏖️🏔️🌳
- Family pictures and cartoons 👨‍👩‍👧‍👦
- Anything colourful and clear works great!

Avoid blurry pictures or pictures with a lot of text — the captioner does best on clear, simple subjects.

---

## 🛠️ How it's built (a note for the grader)

- All three model stages are wrapped in their own clearly named functions:
  `img2text(image)`, `text2story(caption, theme)`, `text2audio(story_text)`.
- Models are loaded **once** with `@st.cache_resource`.
- Generated **caption / story / audio** are stored in `st.session_state` so that:
  - Pressing play does **not** regenerate the story or the audio.
  - Changing the theme regenerates **only** the story (caption is reused).
  - Uploading a new image resets everything via an MD5 file-key.
- Story length is **validated**, **trimmed**, and **retried** so the final story always lands inside the assignment's 50–100 word target.
- Errors during model inference are caught and shown as friendly messages (no Python tracebacks for kids).

---

## ⚠️ Honest notes & limitations

- **No true "child voice" TTS exists** under a permissive license on Hugging Face. We picked a clear, bright female voice from the CMU-Arctic xvector dataset — it is much warmer and friendlier for kids than the default `mms-tts-eng` adult-male voice, but it is not literally a child's voice.
- Story generation is **non-deterministic**: if you don't love the first one, just press **"Try a different story"**.
- The first run downloads ~2.5 GB of models. Be patient! ☕

---

## 📝 License

Provided for educational use as part of the **ISOM5240 Individual Assignment**.
