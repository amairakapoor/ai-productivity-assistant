import streamlit as st
import ffmpeg
import whisper
import spacy
import tempfile
import os
import matplotlib.pyplot as plt
import gc

# ---------------------- UI CONFIG ----------------------
st.set_page_config(page_title="AI Productivity Assistant", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align: center;'>🚀 AI Productivity Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Fast, lightweight AI meeting analyzer</p>", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("⚙️ Settings")
show_tasks = st.sidebar.checkbox("Show Tasks", True)
show_sentiment = st.sidebar.checkbox("Show Sentiment", True)

# ---------------------- FILE UPLOAD ----------------------
st.markdown("### 📂 Upload Audio / Video")
uploaded_file = st.file_uploader("", type=["mp3","wav","mp4","m4a"])

# ---------------------- MODELS ----------------------
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")  # lightweight

    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return whisper_model, nlp

whisper_model, nlp = load_models()

# ---------------------- FFMPEG PATH ----------------------
FFMPEG_PATH = "ffmpeg"

# ---------------------- FUNCTIONS ----------------------
def extract_audio(file_path, output_path="temp_audio.wav"):
    try:
        (
            ffmpeg
            .input(file_path)
            .output(output_path)
            .run(cmd=FFMPEG_PATH, overwrite_output=True)
        )
        return output_path
    except Exception as e:
        st.error(f"FFmpeg Error: {e}")
        return None

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def summarize_text(text):
    sentences = text.split(".")
    return ". ".join(sentences[:2])  # simple lightweight summary

def extract_tasks(text):
    doc = nlp(text)
    tasks = []
    for sent in doc.sents:
        if "will" in sent.text.lower() or "must" in sent.text.lower():
            tasks.append(sent.text.strip())
    return tasks

def analyze_sentiment(text):
    positive_words = ["good", "great", "success", "happy"]
    negative_words = ["bad", "delay", "problem", "issue"]

    pos = sum(word in text.lower() for word in positive_words)
    neg = sum(word in text.lower() for word in negative_words)

    return {"Positive": pos, "Negative": neg}

def plot_sentiment(data):
    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values())
    st.pyplot(fig)

# ---------------------- MAIN ----------------------
if uploaded_file is not None:

    # Save file
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(uploaded_file.read())
    tmp_file.close()

    temp_file_path = tmp_file.name

    # 🔥 Progress UI
    progress = st.progress(0)
    status = st.empty()

    progress.progress(10)
    status.text("📂 File uploaded successfully")

    progress.progress(25)
    status.text("🎧 Extracting audio...")

    audio_path = extract_audio(temp_file_path)

    if audio_path:
        progress.progress(50)
        status.text("🧠 Transcribing audio...")

        text = transcribe_audio(audio_path)
        text = text[:1000]

        progress.progress(70)
        status.text("✍️ Generating summary...")

        summary = summarize_text(text)

        progress.progress(85)
        status.text("📋 Extracting tasks...")

        tasks = extract_tasks(text)

        progress.progress(95)
        status.text("📊 Analyzing sentiment...")

        sentiment = analyze_sentiment(text)

        progress.progress(100)
        status.text("✅ Done!")

        # ---------------- TRANSCRIPTION ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Transcription")
        st.write(text)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- SUMMARY ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📌 Summary")
        st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- TASKS ----------------
        if show_tasks:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### ✅ Tasks")
            for t in tasks:
                st.markdown(f"- {t}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- SENTIMENT ----------------
        if show_sentiment:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Sentiment")
            plot_sentiment(sentiment)
            st.markdown("</div>", unsafe_allow_html=True)

    # Cleanup
    try:
        os.remove(temp_file_path)
        os.remove(audio_path)
    except:
        pass

    gc.collect()
