
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="TuneAI | Music Discovery",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
        radial-gradient(ellipse at 48% 0%, #191438 0%, transparent 48%),
        #080b14;
    color: #eef0ff;
}
[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid #222a40;
}
[data-testid="stSidebar"] * { color: #e7e9fa; }

.brand {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.tagline { color: #929db9; font-size: 12px; margin-top: 3px; }
.hero {
    text-align: center;
    padding: 28px 10px 24px;
}
.hero-icon {
    font-size: 42px;
    margin-bottom: 10px;
}
.hero h1 {
    font-size: clamp(27px, 4vw, 39px);
    letter-spacing: -1.5px;
    font-weight: 800;
    margin: 0;
    color: #f4f1ff;
}
.hero p {
    color: #9da7c3;
    font-size: 14px;
    margin-top: 10px;
}
.section-title {
    font-size: 15px;
    font-weight: 700;
    margin: 8px 0 12px;
}
.panel {
    background: linear-gradient(140deg, #121b2d, #0c1220);
    border: 1px solid #29334d;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 16px;
}
.small-label {
    font-size: 10px;
    letter-spacing: 1.3px;
    color: #a899ff;
    font-weight: 700;
}
.muted { color: #919db8; font-size: 12px; }
div[data-testid="stFileUploader"] {
    border: 1px dashed #6652bb;
    background: #12132a;
    border-radius: 14px;
    padding: 10px;
}
.stButton > button {
    border-radius: 10px;
    border: 1px solid #5441a2;
    background: linear-gradient(110deg, #7048ed, #4e36bb);
    color: white;
    font-weight: 600;
    min-height: 42px;
}
.stButton > button:hover {
    border-color: #ad9aff;
    color: white;
}
div[data-testid="stChatMessage"] {
    background: #111a2a;
    border: 1px solid #253049;
    border-radius: 13px;
    margin-bottom: 10px;
}
div[data-testid="stMetric"] {
    background: #111a2a;
    padding: 12px;
    border: 1px solid #28334c;
    border-radius: 12px;
}
hr { border-color: #252e43; }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="brand">🎧 TuneAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Find the song behind any tune</div>',
        unsafe_allow_html=True
    )
    st.divider()

    if st.button("＋  New detection", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content":
             "Hi! Upload a short audio clip and let's discover its song."}
        ]
        st.session_state.analysis = None
        st.rerun()

    st.markdown("### Navigation")
    st.markdown("🏠  Home")
    st.markdown("🕘  History")
    st.markdown("⚙️  Settings")

    st.divider()
    st.markdown("### ✨ Powered by Gemma 4")
    st.caption("AI-assisted music discovery")
    st.caption("Prototype · Audio engine not connected")

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content":
         "Hi! I'm TuneAI. Upload a short audio clip and I'll help you "
         "discover the song behind it."}
    ]

if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ---------- MAIN LAYOUT ----------
chat_col, analysis_col = st.columns([1.7, 1], gap="large")

with chat_col:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🎶</div>
        <h1>Every song has a signature.</h1>
        <p>Give us 3 seconds. Let's find yours.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💬 Music assistant</div>',
                unsafe_allow_html=True)

    # Chat conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.markdown('<div class="section-title">🎵 Upload your audio</div>',
                unsafe_allow_html=True)

    uploaded_audio = st.file_uploader(
        "Choose an audio clip",
        type=["wav", "mp3", "ogg", "m4a", "flac"],
        help="For the prototype, choose a short clip of around 3 seconds.",
    )

    if uploaded_audio:
        st.audio(uploaded_audio)

        st.success(
            f"Audio selected: {uploaded_audio.name} "
            f"({uploaded_audio.size / 1024:.1f} KB)"
        )

        if st.button("✨ Analyze this song", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": f"Identify this song: {uploaded_audio.name}"
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content":
                "Your audio is uploaded successfully! The interface is ready, "
                "but the FFT engine and song-matching database still need "
                "to be connected. No song has been identified yet."
            })
            st.rerun()

    st.caption(
        "Prototype mode: selecting audio does not yet identify the song."
    )

with analysis_col:
    st.markdown('<div class="section-title">📊 Audio lab</div>',
                unsafe_allow_html=True)

    # Illustrative waveform
    with st.container(border=True):
        st.markdown("**〰️ Waveform**")
        st.caption("Illustrative preview — not extracted from your audio")

        x = np.linspace(0, 6 * np.pi, 120)
        y = np.sin(x) * np.exp(-((x - 3 * np.pi) / 8) ** 2)
        st.line_chart(y, height=140)

    # Illustrative frequency spectrum
    with st.container(border=True):
        st.markdown("**📈 Frequency analysis**")
        st.caption("Illustrative preview — real FFT data will go here")

        frequencies = np.arange(1, 36)
        magnitudes = (
            np.abs(np.sin(frequencies * 0.72)
                   * np.cos(frequencies * 0.21)) * 100
        )
        st.bar_chart(
            {"Magnitude": magnitudes},
            height=170
        )

    with st.container(border=True):
        st.markdown('<div class="small-label">SONG MATCH</div>',
                    unsafe_allow_html=True)
        st.markdown("### 🎼 Your discovery awaits")
        st.write(
            "Upload a clip to begin. Once the backend is connected, "
            "your most likely song matches will appear here."
        )
        st.divider()
        st.markdown("🟢 Audio feature extraction")
        st.markdown("🟡 Song fingerprint matching")
        st.markdown("🟡 Gemma-powered explanation")
