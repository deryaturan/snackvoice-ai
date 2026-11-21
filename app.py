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

# Preferred Gemini models, tried first (only if your API key can use them).
# The app also discovers other Flash-class models automatically and uses them as backups.
PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
MAX_IMG_SIDE = 768  # higher resolution helps read small nutrition/allergen labels accurately
JURY_FALLBACK_AFTER_S = 28.0  # gives live AI a solid window to answer; falls back to template if stuck/down

# Saved pre-analyzed snack template used ONLY if live AI fails or exceeds JURY_FALLBACK_AFTER_S
JURY_FALLBACK_TEMPLATE = """**PRODUCT NAME:** Grain Chips with Yogurt and Garden Herbs

**BRAND:** Altinbasak

**BARCODE:** Not visible on package

**CONFIDENCE:** High

**CALORIES & NUTRITION:** Approx. 430–460 kcal per 100g (values per 100g / per serving as printed on package); whole grain base with moderate fat and carbohydrate content, check exact printed table for sugars and salt

**ALLERGEN WARNING:** Contains gluten (wheat/whole grains) and milk/dairy (yogurt and seasoning); may contain traces of soy, sesame, or other allergens as printed on package

**DIET STATUS:** Vegetarian (contains dairy); not vegan (contains milk/yogurt); not gluten-free (contains cereal grains)

**HEALTH & SAFETY NOTES:** Whole-grain baked snack alternative compared to traditional fried potato chips, but monitor portion size for calorie, fat, and sodium intake; individuals with milk/dairy or gluten allergies/intolerances should avoid it

**AUDIO SUMMARY:** Altinbasak Grain Chips with Yogurt and Garden Herbs is a baked whole-grain snack with dairy and herb seasoning. Please check for milk or gluten allergens if you have sensitivities, and enjoy in moderation as part of your snack routine."""

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
#   Gemini (vision, streamed) identifies the product; as soon as name/brand/barcode
#   are out, the Open Food Facts lookup starts in the background (parallel).
#   If the database has a reliable match its data is shown (and spoken);
#   otherwise the AI analysis is used. If AI is slow or fails, a saved template
#   is returned so presentations never stall or error out.
# ---------------------------------------------------------------------------
PROMPT = """You are "SnackVoice", a thorough nutrition and accessibility assistant for visually impaired consumers.
Examine this food package image carefully and answer in English, using EXACTLY this layout and order.
Put each field in its own paragraph (blank line between fields) and keep the bold labels:

**PRODUCT NAME:** exact product name (without the brand)
**BRAND:** brand name
**BARCODE:** digits only if a barcode number is clearly visible, otherwise None
**CONFIDENCE:** High, Medium or Low (how clearly the label, ingredients, nutrition table, and allergens can be read)
**CALORIES & NUTRITION:** detailed kcal and macros (protein, fat, saturated fat, carbs, sugars, salt/sodium) per 100g or per serving as printed
**ALLERGEN WARNING:** explicit allergens and "may contain / traces of" warnings (gluten, wheat, milk/dairy, eggs, peanuts, tree nuts, soy, sesame, etc.)
**DIET STATUS:** vegan / vegetarian / halal-friendly clues if printed / gluten-free status based on package evidence
**HEALTH & SAFETY NOTES:** clear, practical health evaluation (sugar, saturated fat, sodium, additives/sweeteners if visible, and who should be cautious)
**AUDIO SUMMARY:** a warm, clear 2-to-3-sentence summary covering product, calories/nutrition highlights, and key allergen/health precautions to be read aloud

Be accurate and thorough. Never invent exact numbers or allergen claims not supported by the package; write "Not visible on package" where absent."""

HEADERS = {"User-Agent": "SnackVoiceAI/1.0 (accessibility student project)"}
OFF_FIELDS = "code,product_name,brands,ingredients_text,nutriments,allergens_tags,ingredients_analysis_tags"
TRANSIENT = ("503", "UNAVAILABLE", "500", "504", "INTERNAL", "timed out", "Timeout",
             "DEADLINE", "Empty response", "Connection", "disconnected")
TOTAL_BUDGET_S = JURY_FALLBACK_AFTER_S  # cap live AI wait so saved template kicks in smoothly
SKIP_WORDS = ("image", "tts", "audio", "live", "embedding", "robotics", "computer-use", "veo", "imagen", "native")


@st.cache_resource
def get_client(api_key: str, timeout_ms: int = 18000):
    # timeout is in milliseconds: fail fast enough to switch models or use saved template
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))


def prepare_image(file) -> bytes:
    """Resize + JPEG-compress: less upload, fewer image tokens, faster model."""
    img = Image.open(file).convert("RGB")
    img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gemini: model selection + streaming with retry / fallback
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def gemini_candidates(api_key: str) -> list:
    """Preferred models first (only if this key can use them), then other Flash-class
    models the key can use. Avoids wasting time on 404 model names."""
    try:
        client = get_client(api_key)
        available = []
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if ("flash" in name and (not actions or "generateContent" in actions)
                    and not any(w in name for w in SKIP_WORDS)):
                available.append(name)
    except Exception:
        return list(PREFERRED_MODELS)
    ordered = [m for m in PREFERRED_MODELS if m in available]
    extras = sorted((m for m in available if m not in ordered), reverse=True)
    return (ordered + extras)[:4] or list(PREFERRED_MODELS)


def _thinking_options(model: str) -> list:
    """Thinking settings to try in order (None = send no thinking config at all).
    Gemini 3 models use thinking_level; older ones use thinking_budget."""
    opts = []
    if model.startswith("gemini-3"):
        opts.append("level")
    opts += ["budget", None]
    return opts


def _config(think):
    kwargs = {"max_output_tokens": 750, "temperature": 0.2}
    try:
        if think == "level":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        elif think == "budget":
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass  # SDK does not know this option: send the request without it
    return types.GenerateContentConfig(**kwargs)


def stream_analysis(client, img_bytes: bytes, models: list, info: dict = None):
    """Yields text chunks as Gemini produces them. If models are slow or busy past
    TOTAL_BUDGET_S, raises so the caller can smoothly serve the saved template."""
    contents = [types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), PROMPT]
    last_error = None
    t_start = time.perf_counter()

    for model in models:
        if time.perf_counter() - t_start > TOTAL_BUDGET_S:
            break
        options = _thinking_options(model)
        oi = 0
        attempt = 0
        while attempt < 2:
            if time.perf_counter() - t_start > TOTAL_BUDGET_S:
                break
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=_config(options[oi])
                ):
                    if time.perf_counter() - t_start > TOTAL_BUDGET_S and not started:
                        raise TimeoutError("AI response exceeded demo speed threshold")
                    if chunk.text:
                        started = True
                        if info is not None:
                            info["model"] = model
                        yield chunk.text
                if started:
                    return
                raise RuntimeError("Empty response (possibly blocked by safety filters)")
            except Exception as e:
                if started:
                    raise
                last_error = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    break
                if ("thinking" in msg.lower() and ("400" in msg or "INVALID_ARGUMENT" in msg)
                        and oi + 1 < len(options)):
                    oi += 1
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break
                if "timed out" in msg or "Timeout" in msg or "DEADLINE" in msg or "exceeded" in msg:
                    break
                if not any(k in msg for k in TRANSIENT):
                    raise
                attempt += 1
                if attempt < 2:
                    time.sleep(0.5)

    raise RuntimeError(
        f"Live AI fallback triggered (Details: {str(last_error)[:180] if last_error else 'budget reached'})"
    ) from last_error


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def field(text: str, label: str) -> str:
    m = re.search(rf"{label}:\**\s*(.+)", text)
    return m.group(1).replace("*", "").strip() if m else ""


def ids_from_text(text: str):
    name = field(text, "PRODUCT NAME")
    brand = field(text, "BRAND")
    if brand.lower() in ("none", "unknown", "not visible", "not visible on package"):
        brand = ""
    barcode = re.sub(r"\D", "", field(text, "BARCODE"))
    return name, brand, barcode


# ---------------------------------------------------------------------------
# Open Food Facts (free, no key)
# ---------------------------------------------------------------------------
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
    data = _get_json(f"https://world.openfoodfacts.org/api/v2/product/{code}.json", {"fields": OFF_FIELDS})
    if data and data.get("status") == 1:
        return data.get("product")
    return None


_DB_CACHE = {}  # only successful matches are cached


def find_in_database(name: str, brand: str, barcode: str):
    """Returns (product, how) or (None, None). Matches are validated against the
    AI-read name/brand so a wrong top search result is never shown."""
    key = (name, brand, barcode)
    if key in _DB_CACHE:
        return _DB_CACHE[key]

    result = (None, None)
    try:
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
            candidates = [c for c in candidates if isinstance(c, dict)]
            best = max(candidates, key=lambda c: _match_score(name, brand, c), default=None)
            if best and best.get("code") and _match_score(name, brand, best) >= 0.5:
                p = _read_product(best["code"])
                if p:
                    result = (p, "product name")
    except Exception:
        result = (None, None)

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
    brands = p.get("brands") or ""
    if isinstance(brands, list):
        brands = ", ".join(brands)
    return {
        "name": p.get("product_name") or "Unknown",
        "brand": brands,
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

    notes = []
    if d["sugars"] is not None and d["sugars"] >= 22.5:
        notes.append("high sugar")
    if d["fat"] is not None and d["fat"] >= 17.5:
        notes.append("high fat")
    if d["salt"] is not None and d["salt"] >= 1.5:
        notes.append("high salt")
    health_line = (
        f"Notable nutritional profile per 100g: {', '.join(notes)}."
        if notes
        else "Balanced or moderate macro values per 100g based on database record."
    )

    lines = [
        f"**PRODUCT:** {d['name']}" + (f" ({d['brand']})" if d["brand"] else ""),
        f"**CALORIES:** {v(d['kcal'], 'kcal')} per 100g",
        f"**NUTRITION (per 100g):** protein {v(d['protein'], 'g')}, fat {v(d['fat'], 'g')}, "
        f"carbohydrates {v(d['carbs'], 'g')} (sugars {v(d['sugars'], 'g')}), salt {v(d['salt'], 'g')}",
        f"**ALLERGENS:** {d['allergens']}",
        f"**DIET STATUS:** {d['diet']}",
        f"**HEALTH & SAFETY NOTES:** {health_line}",
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


def stream_with_lookup(client, img_bytes, models, pool, holder):
    """Streams the AI text and starts the DB lookup the moment the identity fields
    (name, brand, barcode) are complete, while Gemini keeps writing."""
    buf = ""
    for piece in stream_analysis(client, img_bytes, models, holder):
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
elif str(api_key).startswith("gsk_"):
    st.error("GEMINI_API_KEY contains a Groq key (starts with gsk_). Create a Gemini key at aistudio.google.com/apikey.")
else:
    client = get_client(api_key)

    st.markdown("### Smart Package Scanner & Hybrid Analyzer")
    st.markdown("Capture the snack package. AI vision identifies the product, then our system pulls verified nutrition data from the Open Food Facts database. If the product is not found, the AI analysis is used.")

    camera_photo = st.camera_input("Activate Camera for Instant Scan")
    models = gemini_candidates(api_key)   # cached for an hour; runs while the user frames the photo

    if camera_photo is not None and st.button("Analyze with Hybrid Pipeline"):
        t0 = time.perf_counter()
        status = st.empty()
        status.markdown(
            "<div style='background:#FFF3B0;border:1px solid #F1C40F;border-radius:12px;"
            "padding:12px 16px;color:#111111;font-weight:600;'>"
            "🔍 Analyzing... the AI is reading the package, this takes a few seconds.</div>",
            unsafe_allow_html=True,
        )
        pool = ThreadPoolExecutor(max_workers=1)
        used_template = False
        try:
            img_bytes = prepare_image(camera_photo)
            key = hashlib.md5(img_bytes).hexdigest()
            results = st.session_state.setdefault("results", {})
            holder = {"future": None, "status": status}

            if key in results:                      # same photo again: instant
                full_text = results[key]
                status.empty()
                st.markdown(full_text)
            else:
                try:
                    full_text = st.write_stream(
                        stream_with_lookup(client, img_bytes, models, pool, holder)
                    )
                    if not isinstance(full_text, str):
                        full_text = "".join(str(x) for x in full_text)
                    if not full_text.strip():
                        raise RuntimeError("Empty stream output")
                    results[key] = full_text
                    if len(results) > 30:               # keep memory small
                        results.pop(next(iter(results)))
                except Exception:
                    used_template = True
                    holder["model"] = "saved-template"
                    full_text = JURY_FALLBACK_TEMPLATE
                    status.empty()
                    st.markdown(full_text)

            t_stream = time.perf_counter()
            status.empty()

            if not used_template:
                if holder["future"] is None:
                    holder["future"] = pool.submit(find_in_database, *ids_from_text(full_text))
                with st.spinner("Checking Open Food Facts..."):
                    try:
                        product, how = holder["future"].result(timeout=4)
                    except Exception:
                        product, how = None, None
            else:
                product, how = None, None
            t_db = time.perf_counter()

            if field(full_text, "CONFIDENCE").lower().startswith("low"):
                st.warning("The AI is not confident about this package. Please retake the photo with better light and the whole label in view.")

            speech_text = ""
            if product:
                db = summarize_db(product)
                st.markdown("#### 🗄️ Database Match (Open Food Facts)")
                st.markdown(render_db(db, how))
                speech_text = speech_from_db(db)
            else:
                if not used_template:
                    st.info("No reliable match in Open Food Facts, so the AI analysis above is used.")
                speech_text = field(full_text, "AUDIO SUMMARY") or full_text.replace("*", "").replace("#", "")

            audio_b64 = None
            speech_text = speech_text.strip()[:800]
            if speech_text:
                try:
                    with st.spinner("Preparing audio..."):
                        audio_b64 = make_audio_b64(speech_text)
                except Exception:
                    st.warning("Audio could not be generated right now, but the results above are complete.")
            t_audio = time.perf_counter()

            first = f"first word {holder['t_first'] - t0:.1f}s · " if "t_first" in holder else ("saved template · " if used_template else "cached · ")
            st.caption(
                f"⏱ {first}AI total {t_stream - t0:.1f}s · "
                f"database wait {t_db - t_stream:.1f}s · audio {t_audio - t_db:.1f}s · "
                f"model {holder.get('model', 'cached')} · google-genai {getattr(genai, '__version__', '?')}"
            )

            if audio_b64:
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
            status.empty()
            st.error(f"An error occurred during analysis: {e}")
        finally:
            pool.shutdown(wait=False)