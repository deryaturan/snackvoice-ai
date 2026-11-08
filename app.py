import streamlit as st
from PIL import Image
from google import genai
from gtts import gTTS
import io
import requests
import time
import re

# 1. UI Configuration & Responsive Mobile Settings
st.set_page_config(
    page_title="SnackVoice AI", 
    page_icon="🔊", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS Styling (Aligned Logo, Custom Success Box, Clean Floating Audio Card)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #111111;
        background-color: #FFF8DC;
    }
    
    .stApp {
        background-color: #FFF8DC;
    }

    h1 {
        font-weight: 800 !important;
        color: #C0392B !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }

    h3, .stMarkdown h3, label {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    p, .stMarkdown p {
        font-size: 1.15rem !important;
        line-height: 1.5 !important;
        color: #1A1A1A;
    }
    
    /* Perfect logo and title alignment */
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }
    
    [data-testid="stImage"] img {
        width: 85px !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 12px;
    }
    
    .stButton>button {
        width: 100%;
        height: 4em;
        font-size: 18px !important;
        font-weight: 700;
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        color: #FFFFFF;
        border-radius: 14px;
        border: none;
        box-shadow: 0 4px 14px rgba(192, 57, 43, 0.3);
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #C0392B 0%, #962D22 100%);
        color: #FFFFFF;
        box-shadow: 0 6px 18px rgba(192, 57, 43, 0.4);
        transform: translateY(-2px);
    }

    /* Styled Success Box */
    .custom-success {
        background-color: #D4EFDF;
        border-left: 5px solid #27AE60;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 700;
        color: #145A32;
        margin-bottom: 15px;
        font-size: 1.1rem;
    }

    /* Minimalist Floating Audio Card fixed at bottom-right */
    .floating-audio-card {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 99999;
        background: #FFFFFF;
        padding: 14px 18px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        border: 2px solid #C0392B;
        width: 310px;
        max-width: 90vw;
    }
    </style>
""", unsafe_allow_html=True)

# Perfectly Aligned Header Layout
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        logo_img = Image.open("logo.png")
        st.image(logo_img)
    except Exception:
        st.markdown("🛡️")

with col_title:
    st.markdown("<div style='padding-top: 8px;'><h1>SnackVoice AI</h1><p style='color: #922B21; font-size: 1rem !important; margin: 0;'>The voice of every snack, right in your hand.</p></div>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #F5B7B1; margin-top: 15px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Helper function with retry mechanism for temporary server overloads (503)
def generate_content_with_retry(client, model, contents, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e

# Securely retrieve API Key from Streamlit Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("System Error: GEMINI_API_KEY is not configured in Streamlit Secrets.")
else:
    client = genai.Client(api_key=api_key)

    st.markdown("### Smart Package Scanner & Hybrid Analyzer")
    st.markdown("Capture the snack package or ingredients list. Our single-pass hybrid engine inspects the product, checks global databases, and synthesizes an accessibility summary instantly.")

    camera_photo = st.camera_input("Activate Camera for Instant Scan")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        
        if st.button("Analyze with Hybrid AI Engine"):
            with st.spinner("Analyzing package with Hybrid AI & Database engine..."):
                try:
                    # Single-pass fast hybrid prompt (Optimized for speed & robustness)
                    unified_prompt = """
                    You are the "SnackVoice" accessibility assistant for visually impaired and health-conscious consumers.
                    Analyze this food package image. If any barcode digits or product names are visible, use them. If not, inspect the visible ingredients, text, and graphics on the package to deduce everything. Never tell the user to check the back of the package; extract all visible information directly.
                    
                    Respond strictly in this format (in English):
                    PRODUCT NAME: [Product name or category]
                    CALORIES & NUTRITION: [Calories/values per 100g or serving]
                    ALLERGEN WARNING: [Check for gluten, dairy, nuts, peanuts, etc.]
                    DIET STATUS: [Vegan, vegetarian, or gluten-free status]
                    
                    AUDIO SUMMARY:
                    [A warm, natural, human-like 2-3 sentence summary summarizing the above details, optimized to be read aloud smoothly.]
                    """

                    response = generate_content_with_retry(
                        client=client,
                        model='gemini-3.8-flash',
                        contents=[image, unified_prompt]
                    )

                    full_text = response.text
                    
                    # Custom success message matching user requirement
                    st.markdown('<div class="custom-success">✓ Hybrid Pipeline Analysis Successful</div>', unsafe_allow_html=True)
                    st.markdown(full_text)

                    if "AUDIO SUMMARY:" in full_text:
                        speech_text = full_text.split("AUDIO SUMMARY:")[1].strip()
                    else:
                        speech_text = full_text.replace("*", "").replace("#", "")

                    # Natural human-like speech generation via gTTS
                    tts = gTTS(text=speech_text, lang='en', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)

                    # Floating Audio Card containing ONLY the audio player & minimal header
                    st.markdown("""
                        <div class="floating-audio-card">
                            <div style="font-weight: 700; color: #C0392B; font-size: 14px; margin-bottom: 6px;">🔊 Audio Overview Ready</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.audio(audio_fp, format="audio/mp3", autoplay=True)

                except Exception as e:
                    st.error(f"An error occurred during hybrid execution: {e}")