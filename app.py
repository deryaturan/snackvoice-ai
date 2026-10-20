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

# 2. Modern Food-Tech Palette & Enhanced Accessibility CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1E292B;
        background-color: #F9F9F6;
    }
    
    .stApp {
        background-color: #F9F9F6;
    }

    h1 {
        font-weight: 800 !important;
        color: #2F5233 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.02em;
    }

    p, .stMarkdown p {
        font-size: 1.125rem !important;
        line-height: 1.6 !important;
        color: #334143;
    }
    
    .stButton>button {
        width: 100%;
        height: 4em;
        font-size: 18px !important;
        font-weight: 700;
        background-color: #2F5233;
        color: #FFFFFF;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(47, 82, 51, 0.2);
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background-color: #3B683F;
        color: #FFFFFF;
        box-shadow: 0 6px 16px rgba(47, 82, 51, 0.3);
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        background-color: #EAECE7;
        border-radius: 10px;
        color: #4A5D4E;
        font-weight: 700;
        font-size: 16px;
        border: none;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2F5233 !important;
        color: #FFFFFF !important;
    }

    .stTextInput input {
        font-size: 18px !important;
        padding: 14px !important;
        border-radius: 10px !important;
        border: 2px solid #D5DBD2 !important;
    }

    /* Custom Large Audio Container */
    .audio-card {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(47, 82, 51, 0.08);
        border: 2px solid #E2E8DE;
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("SnackVoice AI")
st.caption("Smart Food Package & Barcode Assistant for Accessibility")

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

# 2. Securely retrieve API Key from Streamlit Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("System Error: GEMINI_API_KEY is not configured in Streamlit Secrets.")
else:
    client = genai.Client(api_key=api_key)

    st.write("---")
    
    tab1, tab2 = st.tabs(["Package Photo", "Barcode / Product Code"])

    with tab1:
        st.markdown("### Capture Product Package or Ingredients List")
        camera_photo = st.camera_input("Activate Camera")

        if camera_photo is not None:
            image = Image.open(camera_photo)
            
            if st.button("Analyze Image"):
                with st.spinner("Reading contents and preparing audio summary..."):
                    try:
                        prompt = """
                        You are the "SnackVoice" accessibility assistant designed for visually impaired and health-conscious consumers.
                        Analyze the given food package / ingredients image and respond strictly in the following format (in English):

                        PRODUCT NAME: [Product name or category]
                        CALORIES & NUTRITION: [Calories/values per 100g or serving]
                        ALLERGEN WARNING: [Check for gluten, lactose, peanuts, etc.]
                        DIET STATUS: [Vegan, vegetarian, or gluten-free status]
                        
                        AUDIO SUMMARY:
                        [A natural 2-3 sentence summary summarizing the above details, optimized to be read aloud to the user.]
                        """

                        response = generate_content_with_retry(
                            client=client,
                            model='gemini-3.8-flash',
                            contents=[image, prompt]
                        )

                        full_text = response.text
                        st.success("Analysis Completed Successfully")
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
                            <div class="audio-card">
                                <h3 style="color: #2F5233; margin-top: 0; margin-bottom: 10px;">Audio Overview Ready</h3>
                                <p style="font-size: 15px; color: #556B58; margin-bottom: 15px;">Tap play below to listen to the accessibility summary.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.audio(audio_fp, format="audio/mp3", autoplay=True)

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

    with tab2:
        st.markdown("### Enter Product Barcode Number")
        barcode_input = st.text_input("Example: 8690504011234 (Product Barcode)")

        if st.button("Lookup Product in Database"):
            if barcode_input:
                with st.spinner("Querying Open Food Facts database..."):
                    try:
                        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_input}.json"
                        headers = {"User-Agent": "SnackVoiceAI - Accessibility App - Student Project"}
                        res = requests.get(url, headers=headers)

                        if res.status_code == 200:
                            try:
                                data = res.json()
                            except Exception:
                                data = {}

                            if data.get("status") == 1:
                                product = data.get("product", {})
                                product_name = product.get("product_name", "Unknown Product")
                                ingredients = product.get("ingredients_text", product.get("ingredients_text_en", "Ingredients details not available."))
                                nutrients = product.get("nutriments", {})
                                calories = nutrients.get("energy-kcal_100g", "Unknown")

                                db_prompt = f"""
                                This product data was retrieved from the database.
                                Product Name: {product_name}
                                Ingredients: {ingredients}
                                Calories per 100g: {calories} kcal

                                Using this information, generate an accessibility-focused summary for consumers in the following format (in English):
                                PRODUCT NAME: {product_name}
                                CALORIES & NUTRITION: {calories} kcal per 100g
                                ALLERGEN WARNING: [Evaluate ingredients for potential allergens like gluten, dairy, nuts]
                                DIET STATUS: [Evaluate vegan/vegetarian compatibility]
                                
                                AUDIO SUMMARY:
                                [A natural 2-3 sentence audio summary suitable for text-to-speech.]
                                """

                                ai_res = generate_content_with_retry(
                                    client=client,
                                    model='gemini-3.8-flash',
                                    contents=[db_prompt]
                                )

                                full_text = ai_res.text
                                st.success("Database Match Successful")
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
                                    <div class="audio-card">
                                        <h3 style="color: #2F5233; margin-top: 0; margin-bottom: 10px;">Audio Overview Ready</h3>
                                        <p style="font-size: 15px; color: #556B58; margin-bottom: 15px;">Tap play below to listen to the accessibility summary.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.audio(audio_fp, format="audio/mp3", autoplay=True)
                            else:
                                st.warning("Product not found in database for this barcode. Please use the 'Package Photo' tab to scan via vision.")
                        else:
                            st.warning("Database connection failed or returned an invalid status. Please try again.")
                    except Exception as db_err:
                        st.error(f"Database error: {db_err}")
            else:
                st.warning("Please enter a valid barcode number.")