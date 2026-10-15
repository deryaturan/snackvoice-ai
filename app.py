import streamlit as st
from PIL import Image
from google import genai
from gtts import gTTS
import io
import requests

# 1. UI Configuration & Responsive Mobile Settings
st.set_page_config(
    page_title="SnackVoice AI", 
    page_icon="🛡️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
    }
    
    .stButton>button {
        width: 100%;
        height: 3.5em;
        font-size: 16px !important;
        font-weight: 600;
        background-color: #0F172A;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #334155;
        color: #FFFFFF;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: #F1F5F9;
        border-radius: 6px;
        color: #475569;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("SnackVoice AI")
st.caption("Smart Food Package & Barcode Assistant for Accessibility")

# 2. Securely retrieve API Key from Streamlit Secrets (No hardcoding)
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
        st.write("**Capture a Photo of the Product Package or Ingredients List:**")
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

                        response = client.models.generate_content(
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

                        st.write("---")
                        st.write("**Audio Overview:**")
                        st.audio(audio_fp, format="audio/mp3", autoplay=True)

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

    with tab2:
        st.write("**Enter Product Barcode Number (Database Lookup):**")
        barcode_input = st.text_input("Example: 8690504011234 (Product Barcode)")

        if st.button("Lookup Product in Database"):
            if barcode_input:
                with st.spinner("Querying Open Food Facts database..."):
                    try:
                        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_input}.json"
                        res = requests.get(url).json()

                        if res.get("status") == 1:
                            product = res.get("product", {})
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

                            ai_res = client.models.generate_content(
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

                            st.write("---")
                            st.audio(audio_fp, format="audio/mp3", autoplay=True)

                        else:
                            st.warning("Product not found in database for this barcode. Please use the 'Package Photo' tab to scan via vision.")
                    except Exception as db_err:
                        st.error(f"Database connection error: {db_err}")
            else:
                st.warning("Please enter a valid barcode number.")