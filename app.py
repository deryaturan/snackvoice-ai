import streamlit as st
from PIL import Image
from google import genai
from gtts import gTTS
import io
import time
import base64
import requests

# 1. UI Configuration & Page Icon
st.set_page_config(
    page_title="SnackVoice AI", 
    page_icon="🔊", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS Styling (Optimized Logo & Embedded HTML5 Audio Controls in Floating Box)
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
        font-size: 2.6rem !important;
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
    
    [data-testid="stImage"] img {
        width: 170px !important;
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

    /* Fixed Bottom-Right Floating Audio Card Wrapper with Embedded Player */
    .floating-audio-card {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 99999;
        background: #FFFFFF;
        padding: 16px 18px;
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
        border: 2px solid #C0392B;
        width: 320px;
        max-width: 90vw;
    }

    .audio-title {
        font-weight: 800;
        color: #C0392B;
        font-size: 15px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Aligned Header Layout
col_logo, col_title = st.columns([1.2, 3.8])
with col_logo:
    try:
        logo_img = Image.open("logo.png")
        st.image(logo_img)
    except Exception:
        st.markdown("🔊")

with col_title:
    st.markdown("<div style='padding-top: 5px;'><h1>SnackVoice AI</h1><p style='color: #922B21; font-size: 1.05rem !important; margin: 0;'>The voice of every snack, right in your hand.</p></div>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #F5B7B1; margin-top: 15px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Helper function with retry mechanism for temporary server overloads (503)
def generate_content_with_retry(client, model, contents, retries=3, delay=1):
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model, 
                contents=contents,
                config={'max_output_tokens': 800}
            )
        except Exception as e:
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 1.5
            else:
                raise e

# Open Food Facts Database Search Function (DB-First Strategy)
def search_open_food_facts(barcode=None, product_name=None):
    headers = {"User-Agent": "SnackVoiceAI - Accessibility App - Student Project"}
    
    # 1. Try Barcode Search if available
    if barcode and barcode.lower() != "none" and barcode.isdigit():
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        try:
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == 1:
                    return data.get("product", {}), "Database Match (Barcode)"
        except Exception:
            pass

    # 2. Try Product Name Search if available
    if product_name and product_name.lower() != "none":
        search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={requests.utils.quote(product_name)}&json=1"
        try:
            res = requests.get(search_url, headers=headers, timeout=3)
            if res.status_code == 200:
                data = res.json()
                products = data.get("products", [])
                if products:
                    return products[0], "Database Match (Product Name)"
        except Exception:
            pass
            
    return None, None

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
    st.markdown("Capture the snack package. Our system first checks global food databases; if not found, it seamlessly falls back to AI vision analysis.")

    camera_photo = st.camera_input("Activate Camera for Instant Scan")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        image.thumbnail((800, 800)) # Speed optimization
        
        if st.button("Analyze with Hybrid Pipeline"):
            with st.spinner("Step 1: Inspecting package and querying database..."):
                try:
                    # Initial fast vision pass to extract product identity & barcode for DB lookup
                    meta_prompt = """
                    Analyze this food package image. Extract:
                    1. PRODUCT_NAME: [Brand and exact product name]
                    2. BARCODE: [Any visible barcode numbers, or None]
                    Respond strictly in this format:
                    PRODUCT_NAME: ...
                    BARCODE: ...
                    """
                    meta_res = generate_content_with_retry(
                        client=client,
                        model='gemini-3.8-flash',
                        contents=[image, meta_prompt],
                    )
                    meta_text = meta_res.text
                    
                    # Simple text parsing for metadata
                    extracted_name = "Unknown Product"
                    extracted_barcode = None
                    for line in meta_text.split('\n'):
                        if "PRODUCT_NAME:" in line:
                            extracted_name = line.replace("PRODUCT_NAME:", "").strip()
                        elif "BARCODE:" in line:
                            extracted_barcode = line.replace("BARCODE:", "").strip()

                    # Query Database (Open Food Facts)
                    db_product, match_type = search_open_food_facts(barcode=extracted_barcode, product_name=extracted_name)

                    if db_product:
                        # CASE 1: Found in Database -> Enrich with AI formatting
                        db_name = db_product.get("product_name", extracted_name)
                        db_ingredients = db_product.get("ingredients_text", "Ingredients not specified in DB.")
                        db_nutriments = db_product.get("nutriments", {})
                        db_calories = db_nutriments.get("energy-kcal_100g", "Unknown")
                        
                        final_prompt = f"""
                        Source Data: Retrieved from Global Food Database ({match_type}).
                        Product Name: {db_name}
                        Ingredients: {db_ingredients}
                        Calories per 100g: {db_calories} kcal

                        Format the final accessibility summary strictly in English:
                        PRODUCT NAME: {db_name}
                        CALORIES & NUTRITION: {db_calories} kcal per 100g
                        ALLERGEN WARNING: [Evaluate ingredients for allergens like gluten, dairy, nuts]
                        DIET STATUS: [Evaluate vegan, vegetarian compatibility]
                        
                        AUDIO SUMMARY:
                        [A concise, warm 2-sentence summary read aloud smoothly.]
                        """
                        ai_res = generate_content_with_retry(client=client, model='gemini-3.8-flash', contents=[final_prompt])
                    
                    else:
                        # CASE 2: Not in Database -> Fallback to Pure AI Vision Analysis
                        fallback_prompt = f"""
                        The product was not found in external databases. Fallback to AI-native vision analysis.
                        You are the "SnackVoice" accessibility assistant for visually impaired consumers.
                        Analyze this food package image directly and provide complete details.

                        Format strictly in English:
                        PRODUCT NAME: [Product name]
                        CALORIES & NUTRITION: [Calories/values per 100g or serving]
                        ALLERGEN WARNING: [Check for gluten, dairy, nuts, etc.]
                        DIET STATUS: [Vegan, vegetarian, or gluten-free status]
                        
                        AUDIO SUMMARY:
                        [A concise, warm 2-sentence summary read aloud smoothly.]
                        """
                        ai_res = generate_content_with_retry(client=client, model='gemini-3.8-flash', contents=[image, fallback_prompt])

                    full_text = ai_res.text
                    st.markdown(full_text)

                    if "AUDIO SUMMARY:" in full_text:
                        speech_text = full_text.split("AUDIO SUMMARY:")[1].strip()
                    else:
                        speech_text = full_text.replace("*", "").replace("#", "")

                    # Generate Audio and convert to Base64 for embedded HTML player inside floating card
                    tts = gTTS(text=speech_text, lang='en', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    
                    audio_bytes = audio_fp.getvalue()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # Fully embedded floating audio widget with interactive play/pause controls
                    st.markdown(f"""
                        <div class="floating-audio-card">
                            <div class="audio-title">🔊 Audio Overview</div>
                            <audio controls autoplay style="width: 100%; height: 38px;">
                                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")