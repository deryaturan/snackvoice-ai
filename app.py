import base64
import hashlib
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image

# Tried in order: if the first is overloaded (503/429) or missing (404), the next is used.
# Check current model names in Google AI Studio.
MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
MAX_IMG_SIDE = 640  # smaller image = faster upload and fewer tokens (raise to 768 if small text is misread)

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
# Backend: hybrid pipeline
#   Gemini (vision) identifies the product while streaming; as soon as the
#   name/brand/barcode lines are out, the Open Food Facts lookup starts in a
#   background thread, so both run at the same time.
# ---------------------------------------------------------------------------
PROMPT = """You are "SnackVoice", an accessibility assistant for visually impaired consumers.
Look at this food package image and answer in English, using EXACTLY this layout and order.
Put each field in its own paragraph (blank line between fields) and keep the bold labels:

**PRODUCT NAME:** exact product name (without the brand)
**BRAND:** brand name
**BARCODE:** digits only if a barcode number is clearly visible, otherwise None
**CONFIDENCE:** High, Medium or Low (how sure you are about what you can read on the package)
**CALORIES & NUTRITION:** values per 100g or per serving, as printed on the package
**ALLERGEN WARNING:** gluten, dairy, nuts, etc. printed on the package
**DIET STATUS:** vegan / vegetarian / gluten-free status
**AUDIO SUMMARY:** a warm, concise 2-sentence summary to be read aloud

Be brief. Never invent values that are not visible on the package; write "Not visible" instead."""

HEADERS = {"User-Agent": "SnackVoiceAI/1.0 (accessibility student project)"}
OFF_FIELDS = "code,product_name,brands,ingredients_text,nutriments,allergens_tags,ingredients_analysis_tags"


@st.cache_resource
def get_client(api_key: str):
    # timeout is in milliseconds: fail fast instead of hanging on an overloaded model
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=12000))


def prepare_image(file) -> bytes:
    img = Image.open(file).convert("RGB")
    img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


_BAD_MODELS = set()  # models that returned 404 once are skipped for the rest of the session


def stream_analysis(client, img_bytes: bytes):
    """Yields text chunks as Gemini produces them (retry/fallback only before first chunk)."""
    config = types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.2,
        # If a Gemini 3 model rejects this, use:
        # thinking_config=types.ThinkingConfig(thinking_level="minimal")
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    contents = [types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), PROMPT]

    last_error = None
    for model in MODELS:
        if model in _BAD_MODELS:
            continue
        for attempt in range(2):
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                if started:
                    raise
                last_error = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    _BAD_MODELS.add(model)
                    break
                if not any(k in msg for k in ("503", "UNAVAILABLE", "429", "500", "504")):
                    raise
                if attempt == 0:
                    time.sleep(1)
    if last_error is None:
        raise RuntimeError("No available Gemini model. Check the MODELS list at the top of the file.")
    raise RuntimeError(
        "Gemini is very busy right now, please try again in a few seconds."
    ) from last_error


# ---------- text helpers ----------
def field(text: str, label: str) -> str:
    m = re.search(rf"{label}:\**\s*(.+)", text)
    return m.group(1).replace("*", "").strip() if m else ""


def ids_from_text(text: str):
    name = field(text, "PRODUCT NAME")
    brand = field(text, "BRAND")
    if brand.lower() in ("none", "unknown", "not visible"):
        brand = ""
    barcode = re.sub(r"\D", "", field(text, "BARCODE"))
    return name, brand, barcode


# ---------- Open Food Facts ----------
def _tokens(s):
    return {t for t in re.findall(r"[a-z0-9çğıöşü]+", (s or "").lower()) if len(t) >= 3}


def _match_score(ai_name, ai_brand, product) -> float:
    """Share of the AI-read name/brand words that also appear in the DB record."""
    a = _tokens(f"{ai_name} {ai_brand}")
    if not a:
        return 0.0
    brands = product.get("brands") or ""
    if isinstance(brands, list):
        brands = " ".join(brands)
    p = _tokens(f"{product.get('product_name') or ''} {brands}")
    return len(a & p) / len(a)


def _get_json(url, params):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=2.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _read_product(code):
    data = _get_json(
        f"https://world.openfoodfacts.org/api/v2/product/{code}.json", {"fields": OFF_FIELDS}
    )
    if data and data.get("status") == 1:
        return data.get("product")
    return None


_DB_CACHE = {}  # only successful matches are cached


def find_in_database(name: str, brand: str, barcode: str):
    """Returns (product, how) or (None, None). The match is validated against the
    AI-read name/brand so a wrong top search result is never shown."""
    key = (name, brand, barcode)
    if key in _DB_CACHE:
        return _DB_CACHE[key]

    result = (None, None)
    if 8 <= len(barcode) <= 14:
        p = _read_product(barcode)
        if p and _match_score(name, brand, p) >= 0.3:
            result = (p, "barcode")

    query = f"{brand} {name}".strip()
    if result[0] is None and query and name.lower() != "unknown product":
        candidates = []
        data = _get_json("https://search.openfoodfacts.org/search", {"q": query, "page_size": 5})
        if data:
            candidates = data.get("hits") or []
        if not candidates:  # legacy endpoint as a fallback
            data = _get_json(
                "https://world.openfoodfacts.org/cgi/search.pl",
                {"search_terms": query, "search_simple": 1, "action": "process",
                 "json": 1, "page_size": 5, "fields": "code,product_name,brands"},
            )
            if data:
                candidates = data.get("products") or []
        best = max(candidates, key=lambda c: _match_score(name, brand, c), default=None)
        if best and best.get("code") and _match_score(name, brand, best) >= 0.5:
            p = _read_product(best["code"])
            if p:
                result = (p, "product name")

    if result[0] is not None:
        _DB_CACHE[key] = result
    return result


def summarize_db(p: dict) -> dict:
    n = p.get("nutriments") or {}

    def g(k):
        v = n.get(k)
        return round(v, 1) if isinstance(v, (int, float)) else None

    tags = p.get("ingredients_analysis_tags") or []
    if "en:vegan" in tags:
        diet = "Vegan (and vegetarian)"
    elif "en:non-vegetarian" in tags:
        diet = "Not vegetarian, not vegan"
    elif "en:vegetarian" in tags and "en:non-vegan" in tags:
        diet = "Vegetarian, not vegan"
    elif "en:vegetarian" in tags:
        diet = "Vegetarian (vegan status unclear)"
    elif "en:non-vegan" in tags:
        diet = "Not vegan"
    else:
        diet = "Unknown"

    allergens = [t.split(":")[-1].replace("-", " ") for t in (p.get("allergens_tags") or [])]
    return {
        "name": p.get("product_name") or "Unknown",
        "brand": p.get("brands") or "",
        "kcal": g("energy-kcal_100g"),
        "protein": g("proteins_100g"),
        "fat": g("fat_100g"),
        "carbs": g("carbohydrates_100g"),
        "sugars": g("sugars_100g"),
        "salt": g("salt_100g"),
        "allergens": ", ".join(allergens) if allergens else "None listed in the database",
        "diet": diet,
        "ingredients": p.get("ingredients_text") or "",
    }


def render_db(d: dict, how: str) -> str:
    def v(x, unit):
        return f"{x} {unit}" if x is not None else "n/a"

    lines = [
        f"**PRODUCT:** {d['name']} {('(' + d['brand'] + ')') if d['brand'] else ''}",
        f"**CALORIES:** {v(d['kcal'], 'kcal')} per 100g",
        f"**NUTRITION (per 100g):** protein {v(d['protein'], 'g')}, fat {v(d['fat'], 'g')}, "
        f"carbohydrates {v(d['carbs'], 'g')} (sugars {v(d['sugars'], 'g')}), salt {v(d['salt'], 'g')}",
        f"**ALLERGENS:** {d['allergens']}",
        f"**DIET STATUS:** {d['diet']}",
    ]
    if d["ingredients"]:
        lines.append(f"**INGREDIENTS:** {d['ingredients']}")
    lines.append(f"*Verified in Open Food Facts (matched by {how}). Data: Open Food Facts, ODbL.*")
    return "\n\n".join(lines)


def speech_from_db(d: dict) -> str:
    parts = [f"{d['name']}."]
    if d["kcal"] is not None:
        parts.append(f"It has {d['kcal']} kilocalories per 100 grams.")
    parts.append(f"Allergens: {d['allergens']}.")
    parts.append(f"Diet status: {d['diet']}.")
    return " ".join(parts)


def stream_with_lookup(client, img_bytes, pool, holder):
    """Streams the AI text and starts the DB lookup the moment the identity
    fields (name, brand, barcode) are complete, while Gemini keeps writing."""
    buf = ""
    for piece in stream_analysis(client, img_bytes):
        if "t_first" not in holder:
            holder["t_first"] = time.perf_counter()
            if holder.get("status") is not None:
                holder["status"].empty()      # first word arrived: remove the "working" message
        buf += piece
        if holder["future"] is None and "CALORIES" in buf:
            holder["future"] = pool.submit(find_in_database, *ids_from_text(buf))
        yield piece


@lru_cache(maxsize=64)
def make_audio_b64(text: str) -> str:
    tts = gTTS(text=text, lang="en", slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return base64.b64encode(fp.getvalue()).decode("utf-8")


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
    client = get_client(api_key)

    st.markdown("### Smart Package Scanner & Hybrid Analyzer")
    st.markdown("Capture the snack package. AI vision identifies the product, then our system pulls verified nutrition data from the Open Food Facts database. If the product is not found, the AI analysis is used.")

    camera_photo = st.camera_input("Activate Camera for Instant Scan")

    if camera_photo is not None:
        img_bytes = prepare_image(camera_photo)
        key = hashlib.md5(img_bytes).hexdigest()

        if st.button("Analyze with Hybrid Pipeline"):
            try:
                results = st.session_state.setdefault("results", {})
                t0 = time.perf_counter()
                status = st.empty()
                status.info("🔍 Analyzing... the AI is reading the package, this takes a few seconds.")
                st.markdown("#### 🤖 AI Vision Analysis")

                with ThreadPoolExecutor(max_workers=1) as pool:
                    holder = {"future": None, "status": status}
                    if key in results:
                        full_text = results[key]
                        status.empty()
                        st.markdown(full_text)
                    else:
                        full_text = st.write_stream(stream_with_lookup(client, img_bytes, pool, holder))
                        results[key] = full_text

                    t_stream = time.perf_counter()
                    status.empty()
                    if holder["future"] is None:
                        holder["future"] = pool.submit(find_in_database, *ids_from_text(full_text))
                    with st.spinner("Checking Open Food Facts..."):
                        try:
                            product, how = holder["future"].result(timeout=6)
                        except Exception:
                            product, how = None, None
                    t_db = time.perf_counter()

                conf = field(full_text, "CONFIDENCE").lower()
                if conf.startswith("low"):
                    st.warning("The AI is not confident about this package. Please retake the photo with better light and the whole label in view.")

                if product:
                    db = summarize_db(product)
                    st.markdown("#### 🗄️ Database Match (Open Food Facts)")
                    st.markdown(render_db(db, how))
                    speech_text = speech_from_db(db)
                else:
                    st.info("No reliable match in Open Food Facts, so the AI analysis above is used.")
                    speech_text = field(full_text, "AUDIO SUMMARY") or full_text.replace("*", "").replace("#", "")

                with st.spinner("Preparing audio..."):
                    audio_b64 = make_audio_b64(speech_text)
                t_audio = time.perf_counter()
                first = f"first word {holder['t_first'] - t0:.1f}s · " if "t_first" in holder else "cached · "
                st.caption(
                    f"⏱ {first}AI total {t_stream - t0:.1f}s · "
                    f"database wait {t_db - t_stream:.1f}s · audio {t_audio - t_db:.1f}s"
                )

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
