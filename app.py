import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image
from pydantic import BaseModel

# Sırayla denenir: ilki aşırı yüklüyse (503/429) otomatik olarak sonrakine geçilir.
# Yedek model adlarını Google AI Studio'daki güncel listeden kontrol edin.
MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
MAX_IMG_SIDE = 768           # Daha küçük görsel = daha hızlı yükleme + daha az token

# 1. UI Configuration & Page Icon
st.set_page_config(
    page_title="SnackVoice AI",
    page_icon="🔊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 2. Custom CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #111111;
        background-color: #FFF8DC;
    }

    .stApp { background-color: #FFF8DC; }

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
        st.image(Image.open("logo.png"))
    except Exception:
        st.markdown("🔊")

with col_title:
    st.markdown("<div style='padding-top: 5px;'><h1>SnackVoice AI</h1><p style='color: #922B21; font-size: 1.05rem !important; margin: 0;'>The voice of every snack, right in your hand.</p></div>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #F5B7B1; margin-top: 15px; margin-bottom: 20px;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class SnackInfo(BaseModel):
    product_name: str
    barcode: str          # sadece rakamlar, okunamıyorsa "None"
    calories: str
    allergens: str
    diet_status: str
    audio_summary: str


PROMPT = """You are the "SnackVoice" accessibility assistant for visually impaired consumers.
Analyze this food package image and fill every field in English.
- product_name: brand and exact product name
- barcode: digits only if a barcode number is clearly visible, otherwise "None"
- calories: calories/nutrition values per 100g or per serving, as printed
- allergens: gluten, dairy, nuts, etc. found on the package
- diet_status: vegan / vegetarian / gluten-free status
- audio_summary: a concise, warm 2-sentence summary to be read aloud
Be brief. Do not invent values that are not on the package."""


@st.cache_resource
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def prepare_image(file) -> bytes:
    """Küçült + JPEG sıkıştır: ağ üzerinden giden veri ve model işlem süresi azalır."""
    img = Image.open(file).convert("RGB")
    img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_package(img_bytes: bytes, api_key: str) -> dict:
    """TEK Gemini çağrısı. Aynı fotoğraf tekrar gönderilirse anında cache'ten döner."""
    client = get_client(api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SnackInfo,
        max_output_tokens=800,
        temperature=0.2,
        # Düşünme (thinking) kapalı: bu görev için gecikmenin en büyük kaynağı.
        # Gemini 3 serisinde hata verirse bunu şu şekilde değiştirin:
        # thinking_config=types.ThinkingConfig(thinking_level="minimal")
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    contents = [types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), PROMPT]

    last_error = None
    for model in MODELS:
        delay = 1.0
        for attempt in range(2):          # her model için 2 deneme
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                return json.loads(resp.text)
            except Exception as e:
                last_error = e
                transient = any(k in str(e) for k in ("503", "UNAVAILABLE", "429", "500", "504"))
                if not transient:
                    raise                 # gerçek bir hata: yedek modele geçmeyin
                if attempt == 0:
                    time.sleep(delay)
                    delay *= 2
        # Bu model yoğun: döngü bir sonraki yedek modele geçer
    raise RuntimeError(
        "Gemini servisleri şu an çok yoğun, lütfen birkaç saniye sonra tekrar deneyin."
    ) from last_error


@lru_cache(maxsize=64)
def lookup_barcode(barcode: str):
    """Open Food Facts: sadece barkodla, sadece gereken alanlar, kısa timeout."""
    if not (barcode.isdigit() and 8 <= len(barcode) <= 14):
        return None
    try:
        r = requests.get(
            f"https://world.openfoodfacts.org/api/v2/product/{barcode}",
            params={"fields": "product_name,ingredients_text,nutriments"},
            headers={"User-Agent": "SnackVoiceAI - Accessibility App - Student Project"},
            timeout=2,
        )
        if r.status_code == 200 and r.json().get("status") == 1:
            return r.json().get("product", {})
    except Exception:
        pass
    return None


@lru_cache(maxsize=64)
def make_audio_b64(text: str) -> str:
    tts = gTTS(text=text, lang="en", slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return base64.b64encode(fp.getvalue()).decode("utf-8")


def render_result(info: dict, db_product=None) -> str:
    calories = info["calories"]
    extra = ""
    if db_product:
        kcal = (db_product.get("nutriments") or {}).get("energy-kcal_100g")
        if kcal is not None:
            calories = f"{kcal} kcal per 100g (verified: Open Food Facts)"
        if db_product.get("ingredients_text"):
            extra = f"\n\n**INGREDIENTS:** {db_product['ingredients_text']}"
    return (
        f"**PRODUCT NAME:** {info['product_name']}\n\n"
        f"**CALORIES & NUTRITION:** {calories}\n\n"
        f"**ALLERGEN WARNING:** {info['allergens']}\n\n"
        f"**DIET STATUS:** {info['diet_status']}"
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("System Error: GEMINI_API_KEY is not configured in Streamlit Secrets.")
else:
    st.markdown("### Smart Package Scanner & Hybrid Analyzer")
    st.markdown("Capture the snack package. Our system analyzes it with AI vision and verifies nutrition data against global food databases.")

    camera_photo = st.camera_input("Activate Camera for Instant Scan")

    if camera_photo is not None:
        img_bytes = prepare_image(camera_photo)

        if st.button("Analyze with Hybrid Pipeline"):
            try:
                # Adım 1: tek Gemini çağrısı
                with st.spinner("Analyzing package..."):
                    info = analyze_package(img_bytes, api_key)

                # Metni HEMEN göster; kullanıcı sesi beklemeden okuyabilsin
                result_box = st.empty()
                result_box.markdown(render_result(info))

                # Adım 2: veritabanı doğrulaması ve ses üretimi PARALEL
                with ThreadPoolExecutor(max_workers=2) as pool:
                    db_future = pool.submit(lookup_barcode, info["barcode"].strip())
                    audio_future = pool.submit(make_audio_b64, info["audio_summary"])
                    db_product = db_future.result()
                    audio_b64 = audio_future.result()

                if db_product:
                    result_box.markdown(render_result(info, db_product))

                st.markdown(f"""
                    <div class="floating-audio-card">
                        <div class="audio-title">🔊 Audio Overview</div>
                        <audio controls autoplay style="width: 100%; height: 38px;">
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
