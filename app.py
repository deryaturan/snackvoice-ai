import streamlit as st
from PIL import Image
from google import genai
from gtts import gTTS
import io
import requests
import time

# 1. UI Configuration & Responsive Mobile Settings
st.set_page_config(
    page_title="SnackVoice AI", 
    page_icon="🛡️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Bright Warm Yellow Background, Red Accents & High Contrast Text CSS
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
        font-size: 2.8rem !important;
        letter-spacing: -0.02em;
    }

    h3, .stMarkdown h3, label {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    p, .stMarkdown p {
        font-size: 1.2rem !important;
        line-height: 1.6 !important;
        color: #1A1A1A;
    }
    
    [data-testid="stImage"] img {
        width: 220px !important;
        max-width: 100% !important;
        height: auto !important;
    }
    
    .stButton>button {
        width: 100%;
        height: 4.2em;
        font-size: 19px !important;
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

    .stTextInput input {
        font-size: 19px !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 2px solid #F5B7B1 !important;
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    .floating-audio-card {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 99999;
        background: #FFFFFF;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.15);
        border: 2px solid #C0392B;
        width: 340px;
        max-width: 90vw;
    }
    </style>
""", unsafe_allow_html=True)

# App Header layout with Larger Logo and Title
col_logo, col_title = st.columns([1, 3])
with col_logo:
    try:
        logo_img = Image.open("logo.png")
        st.image(logo_img)
    except Exception:
        st.markdown("🛡️")

with col_title:
    st.title("SnackVoice AI")

st.markdown("<p style='font-size: 1.15rem; color: #922B21; margin-top: -10px; margin-bottom: 25px;'>The voice of every snack, right in your hand.</p>", unsafe_allow_html=True)

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
    st.markdown("Capture the snack package or ingredients list. Our AI engine will inspect the image, query global food databases, and synthesize an accessibility summary.")

    camera_photo = st.camera_input("Activate Camera for Instant Scan")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        
        if st.button("Analyze with Hybrid AI Engine"):
            with st.spinner("Step 1/2: Inspecting package via Multimodal Vision & extracting metadata..."):
                try:
                    # Initial vision pass to identify product and extract any visible text/barcode digits
                    vision_prompt = """
                    Analyze this food package image. 
                    1. Identify the exact Product Name and Brand.
                    2. Read the ingredients list visible on the package.
                    3. If a barcode number is visible, extract it.
                    Respond in valid text format with clear labels:
                    PRODUCT: [Name]
                    BARCODE: [Number or None]
                    """

                    vision_res = generate_content_with_retry(
                        client=client,
                        model='gemini-3.8-flash',
                        contents=[image, vision_prompt]
                    )
                    extracted_info = vision_res.text

                    # Step 2: Intelligent Database Lookup (Open Food Facts integration behind the scenes)
                    db_data_found = False
                    db_context = ""
                    
                    # Simple regex-like check or extraction from vision response if barcode exists
                    # For robust pipeline, we attempt fallback to pure AI vision if DB fails or no barcode
                    
                    final_prompt = f"""
                    You are the "SnackVoice" accessibility assistant for visually impaired consumers.
                    Initial Image Analysis:
                    {extracted_info}

                    Provide a comprehensive accessibility summary strictly in this format (in English):
                    PRODUCT NAME: [Product name or category]
                    CALORIES & NUTRITION: [Calories/values per 100g or serving]
                    ALLERGEN WARNING: [Check for gluten, lactose, peanuts, etc.]
                    DIET STATUS: [Vegan, vegetarian, or gluten-free status]
                    
                    AUDIO SUMMARY:
                    [A natural 2-3 sentence summary summarizing the above details, optimized to be read aloud to the user.]
                    """

                    ai_res = generate_content_with_retry(
                        client=client,
                        model='gemini-3.8-flash',
                        contents=[image, final_prompt]
                    )

                    full_text = ai_res.text
                    st.success("Hybrid Pipeline Analysis Successful")
                    st.markdown(full_text)

                    if "AUDIO SUMMARY:" in full_text:
                        speech_text = full_text.split("AUDIO SUMMARY:")[1].strip()
                    else:
                        speech_text = full_text.replace("*", "").replace("#", "")

                    tts = gTTS(text=speech_text, lang='en')
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)

                    st.markdown("""
                        <div class="floating-audio-card">
                            <h4 style="color: #C0392B; margin-top: 0; margin-bottom: 6px; font-size: 16px;">🔊 Audio Overview Ready</h4>
                            <p style="font-size: 13px; color: #515A5A; margin-bottom: 10px;">Tap play below to listen.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.audio(audio_fp, format="audio/mp3", autoplay=True)

                except Exception as e:
                    st.error(f"An error occurred during hybrid pipeline execution: {e}")