"""
RouteGuide: AI-Powered Smart Multi-Stop Itinerary & Transit Planner
System Analysis and Design project (RG) - Streamlit prototype

Implements the report requirements:
  UC-01 Set starting location (browser GPS -> saved spots -> manual search)
  UC-02 Select stops (max. 5, BR-01)
  UC-03 AI popular pick (Gemini + Google Places verification)
  UC-04 Generate multi-stop itinerary (pairwise legs, Python sequencing, cache)
  UC-05 Compare transit vs driving (same stop order)
  UC-06 Personalized daily plan (interests, hours, pace, budget)
  UC-07 Open route in Google Maps (external export link)
Stateless: everything lives in st.session_state, nothing is stored (NFR-04).
"""
from __future__ import annotations

import copy
import html
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime, timedelta, timezone
from itertools import permutations
from urllib.parse import urlencode

import requests
import streamlit as st
from streamlit_js_eval import get_geolocation

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MAX_STOPS = 5                 # BR-01
MAX_RADIUS_KM = 15            # candidate venues must be near the anchor point
HTTP_TIMEOUT = 12
DRIVE_TRAFFIC_AWARE = True    # set False to save Routes API cost

# Pre-configured saved spots (resolved through Places when used). Edit freely.
PRESET_SPOTS = {
    "City Center (Konak Square)": "Konak Square, Izmir, Turkiye",
    "Ege University Metro Station": "Ege Universitesi Metro Istasyonu, Izmir",
    "Alsancak Train Station": "Alsancak Gari, Izmir",
    "Clock Tower, Konak": "Izmir Clock Tower, Konak, Izmir",
}

# label -> (Places query, default stay in minutes, best time of day)
CATEGORIES = {
    "☕ Café": {"q": "best cafe", "stay": 45, "slot": "any"},
    "🍽️ Restaurant": {"q": "restaurant", "stay": 60, "slot": "midday"},
    "🏛️ Museum": {"q": "museum", "stay": 90, "slot": "morning"},
    "🌳 Park & nature": {"q": "park", "stay": 60, "slot": "any"},
    "🕌 Historical site": {"q": "historical landmark", "stay": 60, "slot": "morning"},
    "🛍️ Shopping": {"q": "shopping", "stay": 60, "slot": "afternoon"},
    "🍰 Dessert": {"q": "dessert shop", "stay": 30, "slot": "afternoon"},
    "🌅 Viewpoint": {"q": "scenic viewpoint", "stay": 30, "slot": "evening"},
    "🍸 Bar": {"q": "bar", "stay": 60, "slot": "evening"},
}
SLOTS = {"any": "Any time", "morning": "Morning", "midday": "Midday",
         "afternoon": "Afternoon", "evening": "Evening"}
SLOT_WINDOWS = {"morning": (6 * 60, 12 * 60), "midday": (11 * 60 + 30, 14 * 60 + 30),
                "afternoon": (12 * 60, 18 * 60), "evening": (17 * 60, 23 * 60)}
VISIT_OPTIONS = [15, 30, 45, 60, 90, 120, 150, 180, 240]
DEFAULT_STAY = 45
PALETTE = ["#7c3aed", "#ec4899", "#f59e0b", "#10b981", "#3b82f6"]
DEFAULT_PLAN_QUERIES = [
    "popular tourist attraction", "best restaurant", "best cafe", "museum", "park",
]
PACE_TARGET = {"Relaxed": "2-3", "Moderate": "3-4", "Packed": "4-5"}
BUDGETS = ["No preference", "Budget-friendly", "Mid-range", "Premium"]

FARE_DISCLAIMER = (
    "Fares are estimates returned by the Google Maps APIs. Actual fares may vary "
    "according to real-time municipal updates or ticket types."
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("routeguide")

st.set_page_config(
    page_title="RouteGuide",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


MAPS_KEY = _secret("GOOGLE_MAPS_API_KEY")
GEMINI_KEY = _secret("GEMINI_API_KEY")
GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
# If the chosen model returns 404 (retired / not available to this project),
# the next ones are tried automatically.
GEMINI_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]

# --------------------------------------------------------------------------
# Styling (mobile first, works down to 375px, no horizontal scrolling)
# --------------------------------------------------------------------------
CSS = """
<style>
.block-container{padding:4.2rem .9rem 4rem!important;max-width:760px}
/* hero */
.hero{border-radius:22px;padding:1.2rem 1.2rem 1rem;color:#fff;margin-bottom:1rem;
  background:linear-gradient(135deg,#7c3aed 0%,#ec4899 55%,#f59e0b 100%);
  box-shadow:0 8px 24px rgba(124,58,237,.28)}
.hero .t{font-size:2rem;font-weight:900;line-height:1.1;letter-spacing:-.5px}
.hero .s{opacity:.95;margin:.25rem 0 .7rem;font-size:.98rem}
.hero .chips span{display:inline-block;background:rgba(255,255,255,.22);border-radius:999px;
  padding:.15rem .6rem;margin:0 .3rem .3rem 0;font-size:.78rem;font-weight:600}
/* step headers */
.step{display:flex;align-items:center;gap:.6rem;margin:.6rem 0 .7rem;font-size:1.25rem;font-weight:800}
.step .n{flex:none;width:2.1rem;height:2.1rem;border-radius:50%;display:grid;place-items:center;
  color:#fff;font-weight:800;box-shadow:0 3px 10px rgba(0,0,0,.18)}
/* keep small button rows side by side on phones */
div[data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:.4rem!important}
div[data-testid="stColumn"],div[data-testid="column"]{min-width:0!important;flex:1 1 0!important;width:auto!important}
.stButton>button,.stLinkButton>a,.stDownloadButton>button{width:100%}
[data-testid^="stBaseButton-secondary"],[data-testid="stBaseLinkButton-secondary"]{border-radius:12px!important}
button[kind="primary"],[data-testid="stBaseButton-primary"],[data-testid="stBaseLinkButton-primary"]{
  background:linear-gradient(135deg,#7c3aed,#ec4899)!important;border:none!important;color:#fff!important;
  font-weight:700!important;border-radius:12px!important;box-shadow:0 4px 14px rgba(236,72,153,.3)}
[data-testid="stBaseButton-secondaryFormSubmit"]{background:linear-gradient(135deg,#0ea5e9,#10b981)!important;
  color:#fff!important;border:none!important;font-weight:700!important}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#7c3aed,#ec4899)!important;height:3px!important}
button[data-baseweb="tab"]{font-weight:700}
/* cards */
.sub{opacity:.72;font-size:.85rem;word-break:break-word}
.card{display:flex;gap:.7rem;align-items:flex-start;border:1px solid rgba(128,128,128,.25);
  border-left:6px solid var(--c);border-radius:14px;padding:.6rem .8rem;margin-bottom:.45rem;
  background:linear-gradient(90deg,color-mix(in srgb,var(--c) 12%,transparent),transparent)}
.card .bd{flex:none;min-width:1.8rem;height:1.8rem;border-radius:50%;background:var(--c);color:#fff;
  display:grid;place-items:center;font-weight:800;font-size:.9rem}
.card .tx{min-width:0;flex:1}
.startcard{border-radius:16px;padding:.8rem 1rem;color:#fff;margin-bottom:.5rem;
  background:linear-gradient(135deg,#10b981,#0ea5e9)}
.startcard .sub{color:#fff;opacity:.9}
.pips{display:flex;gap:.35rem;align-items:center;margin:.2rem 0 .7rem}
.pip{width:100%;max-width:56px;height:10px;border-radius:999px;background:rgba(128,128,128,.25)}
.pip.on{background:var(--c)}
.pips .cnt{margin-left:.4rem;font-weight:700;font-size:.85rem;white-space:nowrap}
.daycard{border-radius:16px;padding:.8rem 1rem;margin:.2rem 0 .8rem;
  background:linear-gradient(135deg,rgba(124,58,237,.14),rgba(236,72,153,.14));
  border:1px solid rgba(124,58,237,.35)}
/* comparison + fare */
.cmp{display:flex;gap:.6rem;flex-wrap:wrap;margin:.4rem 0 .8rem}
.cmp .box{flex:1 1 140px;border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:.7rem .8rem}
.cmp .box.win{border:2px solid #10b981;background:rgba(16,185,129,.1)}
.cmp .lbl{font-size:.85rem;opacity:.8}
.cmp .val{font-size:1.4rem;font-weight:800;line-height:1.3}
.cmp .badge{display:inline-block;margin-top:.2rem;font-size:.7rem;font-weight:700;
  padding:.05rem .45rem;border-radius:999px;background:#10b981;color:#fff}
.fare{border:2px dashed #f59e0b;border-radius:14px;padding:.6rem .8rem;margin-bottom:.6rem;
  background:rgba(245,158,11,.08)}
.fare b{font-size:1.2rem}
/* agenda timeline */
.tl{margin:.4rem 0 1rem}
.tl-row{display:flex;gap:.55rem}
.tl-time{flex:none;width:3.1rem;text-align:right;font-weight:800;font-size:.92rem;
  font-variant-numeric:tabular-nums;padding-top:.05rem}
.tl-rail{flex:none;width:16px;display:flex;flex-direction:column;align-items:center}
.tl-dot{flex:none;width:14px;height:14px;border-radius:50%;background:var(--c);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--c) 22%,transparent);margin-top:.15rem}
.tl-dot.sm{width:8px;height:8px;margin-top:.3rem;box-shadow:none;opacity:.7}
.tl-line{flex:1;width:3px;min-height:12px;background:linear-gradient(var(--c),rgba(128,128,128,.25))}
.tl-body{flex:1;min-width:0;padding-bottom:.65rem}
.tl-card{border-radius:14px;padding:.55rem .8rem;border:1px solid rgba(128,128,128,.25);
  border-left:6px solid var(--c);
  background:linear-gradient(90deg,color-mix(in srgb,var(--c) 12%,transparent),transparent)}
.tl-card .ttl{font-weight:800}
.tl-travel{font-size:.84rem;opacity:.85;padding:.1rem 0 .1rem .1rem}
.tl-end{font-weight:800}
</style>
"""

HERO = """
<div class="hero"><div class="t">🧭 RouteGuide</div>
<div class="s">Plan your day, hop between stops, see the fare.</div>
<div class="chips"><span>📍 Locate</span><span>✨ AI picks</span><span>🗓️ Timed agenda</span>
<span>🚌 Transit vs 🚗 Drive</span><span>🗺️ Maps export</span></div></div>
"""


def step_header(n: int, title: str, color: str):
    st.markdown(f"<div class='step'><span class='n' style='background:{color}'>{n}</span>"
                f"{title}</div>", unsafe_allow_html=True)


def card_html(badge: str, name: str, lines: list[str], color: str) -> str:
    sub = "".join(f"<div class='sub'>{safe(x)}</div>" for x in lines if x)
    return (f"<div class='card' style='--c:{color}'><span class='bd'>{badge}</span>"
            f"<div class='tx'><b>{safe(name)}</b>{sub}</div></div>")


# --------------------------------------------------------------------------
# Session state (stateless app: nothing leaves st.session_state, NFR-04)
# --------------------------------------------------------------------------
DEFAULTS = dict(
    start=None, stops=[], saved={},
    gps_state="pending", gps_candidate=None,
    start_results=[], start_search_no=0,
    stop_results=[], stop_search_no=0,
    ai_picks=[], ai_idx=0,
    plan=None, plan_no=0,
    result=None, gen_error=None,
    optimize=True, day_hours=None, day_summary="", auto_generate=False,
    cache={}, flash={},
)


def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = copy.deepcopy(v)


ss = st.session_state


def flash(channel: str, kind: str, msg: str):
    ss.flash[channel] = (kind, msg)


def show_flash(channel: str):
    item = ss.flash.pop(channel, None)
    if item:
        getattr(st, item[0])(item[1])


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def safe(text) -> str:
    """Escape external text for markdown/HTML ($ would start LaTeX)."""
    return html.escape(str(text or "")).replace("$", "&#36;")


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fmt_dur(sec: int) -> str:
    m = round(sec / 60)
    if m < 1:
        return "< 1 min"
    h, m = divmod(m, 60)
    if h and m:
        return f"{h} h {m} min"
    return f"{h} h" if h else f"{m} min"


CURRENCY = {"TRY": "₺", "USD": "$", "EUR": "€", "GBP": "£"}


def fmt_money(amount: float, cur: str) -> str:
    sym = CURRENCY.get(cur)
    return f"{sym}{amount:,.2f}" if sym else f"{amount:,.2f} {cur}"


def snap(m) -> int:
    return min(VISIT_OPTIONS, key=lambda o: abs(o - int(m or DEFAULT_STAY)))


def slot_penalty(slot: str, minute: int) -> int:
    w = SLOT_WINDOWS.get(slot)
    return max(0, w[0] - minute, minute - w[1]) if w else 0


def maps_url(origin: dict, dests: list[dict], mode: str) -> str:
    p = {
        "api": "1",
        "origin": f"{origin['lat']},{origin['lng']}",
        "destination": f"{dests[-1]['lat']},{dests[-1]['lng']}",
        "travelmode": mode,
    }
    if len(dests) > 1:
        p["waypoints"] = "|".join(f"{d['lat']},{d['lng']}" for d in dests[:-1])
    return "https://www.google.com/maps/dir/?" + urlencode(p)


# --------------------------------------------------------------------------
# HTTP + error handling
# --------------------------------------------------------------------------
class ApiError(Exception):
    pass


class QuotaError(ApiError):
    pass


def friendly(e: Exception) -> str:
    if isinstance(e, QuotaError):
        return ("The service quota or rate limit was reached. Your inputs and the "
                "results already retrieved are kept; please retry in a moment.")
    return f"{e} Your inputs are kept; please retry."


def _http(method: str, url: str, **kw):
    last = None
    for attempt in range(2):
        try:
            r = requests.request(method, url, timeout=HTTP_TIMEOUT, **kw)
        except requests.RequestException:
            last = ApiError("Network error while contacting the service.")
            continue
        if r.status_code in (500, 502, 503, 504) and attempt == 0:
            time.sleep(0.4)
            continue
        return r
    raise last or ApiError("The service is unavailable.")


def _raise_for(r, what: str):
    try:
        body = r.json()
        err = body.get("error", {}) if isinstance(body, dict) else {}
    except ValueError:
        err = {}
    status = err.get("status", "")
    msg = err.get("message", "")[:220]
    if r.status_code == 429 or status == "RESOURCE_EXHAUSTED":
        raise QuotaError(f"{what}: quota exceeded.")
    if r.status_code in (400, 401, 403):
        raise ApiError(f"{what}: request rejected ({status or r.status_code}). {msg}")
    raise ApiError(f"{what}: service error ({r.status_code}).")


# --------------------------------------------------------------------------
# Google Places (New) / Geocoding / Routes
# --------------------------------------------------------------------------
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS_LEAN = "places.id,places.displayName,places.formattedAddress,places.location"
FIELDS_RICH = (FIELDS_LEAN + ",places.rating,places.userRatingCount,places.priceLevel,"
               "places.googleMapsUri,places.primaryTypeDisplayName")
PRICE = {"PRICE_LEVEL_FREE": "Free", "PRICE_LEVEL_INEXPENSIVE": "$",
         "PRICE_LEVEL_MODERATE": "$$", "PRICE_LEVEL_EXPENSIVE": "$$$",
         "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$"}


def _norm_place(p: dict):
    loc = p.get("location") or {}
    if p.get("id") is None or loc.get("latitude") is None:
        return None
    return {
        "id": p["id"],
        "name": (p.get("displayName") or {}).get("text", "Unnamed place"),
        "address": p.get("formattedAddress", ""),
        "lat": loc["latitude"], "lng": loc["longitude"],
        "rating": p.get("rating"), "reviews": p.get("userRatingCount"),
        "price": PRICE.get(p.get("priceLevel")),
        "type": (p.get("primaryTypeDisplayName") or {}).get("text"),
        "uri": p.get("googleMapsUri"),
    }


def places_search(query: str, bias: dict | None = None, radius_m: int = 20000,
                  n: int = 5, rich: bool = False) -> list[dict]:
    cache = ss.cache.setdefault("places", {})
    key = (query.lower(), round(bias["lat"], 3) if bias else None,
           round(bias["lng"], 3) if bias else None, radius_m, n, rich)
    if key in cache:
        return cache[key]
    body = {"textQuery": query, "pageSize": n, "languageCode": "en"}
    if bias:
        body["locationBias"] = {"circle": {
            "center": {"latitude": bias["lat"], "longitude": bias["lng"]},
            "radius": float(radius_m)}}
    r = _http("POST", PLACES_URL, json=body, headers={
        "Content-Type": "application/json", "X-Goog-Api-Key": MAPS_KEY,
        "X-Goog-FieldMask": FIELDS_RICH if rich else FIELDS_LEAN})
    if r.status_code != 200:
        _raise_for(r, "Places search")
    out = [p for p in (_norm_place(x) for x in r.json().get("places", [])) if p]
    cache[key] = out
    return out


def reverse_geocode(lat: float, lng: float):
    r = _http("GET", "https://maps.googleapis.com/maps/api/geocode/json",
              params={"latlng": f"{lat},{lng}", "key": MAPS_KEY, "language": "en"})
    data = r.json()
    status = data.get("status")
    if status == "OK" and data.get("results"):
        return data["results"][0]["formatted_address"]
    if status == "ZERO_RESULTS":
        return None
    if status == "OVER_QUERY_LIMIT":
        raise QuotaError("Geocoding: quota exceeded.")
    raise ApiError(f"Geocoding: {status} {data.get('error_message', '')[:160]}")


ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


def fetch_leg(a: dict, b: dict, mode: str, key: str) -> dict:
    """One pairwise leg. Pure function (thread-safe, no Streamlit state)."""
    if haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]) < 0.03:
        return {"status": "ok", "seconds": 0, "fare": None}
    transit = mode == "TRANSIT"
    body = {
        "origin": {"location": {"latLng": {"latitude": a["lat"], "longitude": a["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": b["lat"], "longitude": b["lng"]}}},
        "travelMode": mode, "languageCode": "en-US", "units": "METRIC",
    }
    if not transit and DRIVE_TRAFFIC_AWARE:
        body["routingPreference"] = "TRAFFIC_AWARE"
    mask = "routes.duration,routes.travelAdvisory.transitFare" if transit else "routes.duration"
    r = _http("POST", ROUTES_URL, json=body, headers={
        "Content-Type": "application/json", "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": mask})
    if r.status_code == 404:
        return {"status": "no_route"}
    if r.status_code != 200:
        _raise_for(r, "Routes API")
    routes = r.json().get("routes") or []
    if not routes or not routes[0].get("duration"):
        return {"status": "no_route"}
    seconds = int(float(str(routes[0]["duration"]).rstrip("s")))
    fare = None
    if transit:
        tf = (routes[0].get("travelAdvisory") or {}).get("transitFare")
        if tf and tf.get("currencyCode"):
            amount = int(tf.get("units", 0) or 0) + (tf.get("nanos", 0) or 0) / 1e9
            fare = {"amount": amount, "currency": tf["currencyCode"]}
    return {"status": "ok", "seconds": seconds, "fare": fare}


def leg_key(a, b, mode):
    return f"{a['lat']:.5f},{a['lng']:.5f}>{b['lat']:.5f},{b['lng']:.5f}|{mode}"


def ensure_legs(pairs: list[tuple[dict, dict]], mode: str) -> list[dict]:
    """Leg-level session cache (NFR-01): only uncached origin-destination-mode
    pairs are requested, in parallel to stay within the 3 s target (NFR-03)."""
    cache = ss.cache.setdefault("legs", {})
    todo, seen = [], set()
    for a, b in pairs:
        k = leg_key(a, b, mode)
        if k not in cache and k not in seen:
            todo.append((k, a, b))
            seen.add(k)
    errors = []
    if todo:
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            futures = [(k, ex.submit(fetch_leg, a, b, mode, MAPS_KEY)) for k, a, b in todo]
            for k, f in futures:
                try:
                    cache[k] = f.result()
                except ApiError as e:
                    errors.append(e)
    if errors:
        raise next((e for e in errors if isinstance(e, QuotaError)), errors[0])
    return [cache[leg_key(a, b, mode)] for a, b in pairs]


# --------------------------------------------------------------------------
# Gemini (recommendation engine only; never used for routing)
# --------------------------------------------------------------------------
def gemini_json(prompt: str, system: str, temperature: float = 0.4):
    if not GEMINI_KEY:
        raise ApiError("Gemini API key is not configured.")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature,
                             "responseMimeType": "application/json"},
    }
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    r = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = _http("POST", url, json=body, headers={
            "Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY})
        if r.status_code != 404:
            break
        log.warning("Gemini model %s returned 404, trying next", model)
    if r.status_code != 200:
        if r.status_code == 404:
            raise ApiError("Gemini: none of the configured models is available to this key. "
                           "Set GEMINI_MODEL in the app secrets.")
        _raise_for(r, "Gemini")
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except (KeyError, IndexError, ValueError):
        raise ApiError("Gemini returned an unusable answer.")


REC_SYSTEM = (
    "You are the recommendation engine of RouteGuide, a travel itinerary planner. "
    "You receive JSON with a traveler's context and a list of VERIFIED candidate venues "
    "from Google Places. Choose ONLY from the candidates and refer to them by their "
    "'cid'. Never invent venues. Prefer strong ratings with many reviews, fit with the "
    "traveler's preferences, and convenience relative to the start and existing stops. "
    "Write every rationale in 1-2 plain English sentences. Respond with JSON only."
)


def gather_candidates(queries: list[str], anchor: dict, limit: int = 12) -> list[dict]:
    """Google Places candidates near the anchor, excluding places already used."""
    exclude = {s["id"] for s in ss.stops} | {ss.start["id"]}
    found: dict[str, dict] = {}
    for q in queries:
        for p in places_search(q, bias=anchor, radius_m=int(MAX_RADIUS_KM * 1000),
                               n=15, rich=True):
            if p["id"] in exclude or p["id"] in found:
                continue
            km = haversine_km(anchor["lat"], anchor["lng"], p["lat"], p["lng"])
            if km > MAX_RADIUS_KM:
                continue
            p = dict(p, km=round(km, 1))
            p["score"] = (p["rating"] or 0) * math.log10((p["reviews"] or 0) + 1)
            found[p["id"]] = p
    ranked = sorted(found.values(), key=lambda x: x["score"], reverse=True)[:limit]
    for i, p in enumerate(ranked, 1):
        p["cid"] = f"c{i}"
    return ranked


def candidate_payload(cands: list[dict]) -> list[dict]:
    return [{"cid": c["cid"], "name": c["name"], "type": c["type"], "rating": c["rating"],
             "reviews": c["reviews"], "price": c["price"], "address": c["address"],
             "km_from_anchor": c["km"]} for c in cands]


def ai_popular_picks(category_label: str, prefs: str) -> list[dict]:
    """UC-03: returns ranked, Places-verified picks (never unverified ones)."""
    anchor = ss.stops[-1] if ss.stops else ss.start
    cands = gather_candidates([CATEGORIES[category_label]["q"]], anchor)
    if not cands:
        return []
    prompt = json.dumps({
        "task": f"Pick the 3 best '{category_label}' venues, best first.",
        "traveler_preferences": prefs or "none given",
        "start": ss.start["address"] or ss.start["name"],
        "existing_stops": [s["name"] for s in ss.stops],
        "candidates": candidate_payload(cands),
        "output_format": {"picks": [{"cid": "<candidate cid>", "rationale": "<why>"}]},
    }, ensure_ascii=False)
    data = gemini_json(prompt, REC_SYSTEM)
    cmap = {c["cid"]: c for c in cands}
    picks = []
    for item in (data.get("picks") if isinstance(data, dict) else None) or []:
        c = cmap.get(str(item.get("cid")))
        if c and all(c["id"] != p["id"] for p in picks):
            picks.append(dict(c, rationale=str(item.get("rationale", ""))[:400],
                              visit_minutes=CATEGORIES[category_label]["stay"],
                              slot=CATEGORIES[category_label]["slot"]))
    return picks


def daily_plan(interests: str, hours: int, pace: str, budget: str) -> dict:
    """UC-06: preference parsing (Gemini) -> Places candidates -> selection (Gemini)."""
    anchor = ss.start
    notice = None
    queries = DEFAULT_PLAN_QUERIES
    if interests.strip():
        data = gemini_json(json.dumps({
            "task": "Convert the traveler's interests into at most 4 short Google Places "
                    "text-search queries (max 5 words each), diverse and specific.",
            "interests": interests.strip(),
            "output_format": {"queries": ["..."]}}, ensure_ascii=False),
            "You turn travel interests into search queries. Respond with JSON only.")
        qs = [str(q) for q in (data.get("queries") if isinstance(data, dict) else None) or []]
        queries = qs[:4] or DEFAULT_PLAN_QUERIES
    else:
        notice = "No preferences specified: a general, category-balanced plan was used."
    cands = gather_candidates(queries, anchor, limit=15)
    if not cands:
        return {"picks": [], "summary": "", "notice": notice, "hours": hours}
    prompt = json.dumps({
        "task": "Build a one-day plan from the candidates. Choose between 1 and 5 venues. "
                "Visiting order does not matter (routing is handled elsewhere).",
        "interests": interests.strip() or "none given (balance categories)",
        "available_hours": hours,
        "pace": pace, "target_number_of_venues_for_pace": PACE_TARGET[pace],
        "budget": budget,
        "rules": "Total visit_minutes must fit within available hours, leaving about 20% "
                 "for travel. Respect the budget using each candidate's price level. Give every pick a 'slot' chosen from: morning, midday, afternoon, evening, any (e.g. museums in the morning, restaurants at midday or evening, viewpoints near sunset, bars in the evening).",
        "start": anchor["address"] or anchor["name"],
        "existing_stops": [s["name"] for s in ss.stops],
        "candidates": candidate_payload(cands),
        "output_format": {"summary": "<2 sentence overview>",
                          "picks": [{"cid": "<cid>", "rationale": "<why>",
                                     "visit_minutes": 60, "slot": "midday"}]},
    }, ensure_ascii=False)
    data = gemini_json(prompt, REC_SYSTEM)
    cmap = {c["cid"]: c for c in cands}
    picks = []
    for item in (data.get("picks") if isinstance(data, dict) else None) or []:
        c = cmap.get(str(item.get("cid")))
        if not c or any(c["id"] == p["id"] for p in picks):
            continue
        try:
            vm = max(15, min(240, int(item.get("visit_minutes", 60))))
        except (TypeError, ValueError):
            vm = 60
        slot = str(item.get("slot", "any")).lower()
        picks.append(dict(c, rationale=str(item.get("rationale", ""))[:400],
                          visit_minutes=snap(vm), slot=slot if slot in SLOTS else "any"))
    summary = str(data.get("summary", ""))[:500] if isinstance(data, dict) else ""
    return {"picks": picks[:5], "summary": summary, "notice": notice, "hours": hours}


# --------------------------------------------------------------------------
# Stop management (UC-02, BR-01)
# --------------------------------------------------------------------------
def add_stop(place: dict) -> tuple[bool, str]:
    if not ss.start:
        return False, "Set a starting point before adding stops."
    if len(ss.stops) >= MAX_STOPS:
        return False, (f"Limit reached: a maximum of {MAX_STOPS} stops is allowed. "
                       "Remove a stop to add another.")
    for s in ss.stops:
        if s["id"] == place["id"] or haversine_km(s["lat"], s["lng"], place["lat"], place["lng"]) < 0.03:
            return False, f"“{place['name']}” is already in your itinerary."
    keep = ("id", "name", "address", "lat", "lng", "rating", "reviews", "price", "type",
            "uri", "rationale")
    stop = {k: place.get(k) for k in keep}
    stop["visit_minutes"] = snap(place.get("visit_minutes") or DEFAULT_STAY)
    stop["slot"] = place.get("slot") if place.get("slot") in SLOTS else "any"
    ss.stops.append(stop)
    return True, f"Added “{place['name']}”."


def cb_add_stop(place: dict, clear_ai: bool = False):
    ok, msg = add_stop(place)
    flash("stops", "success" if ok else "error", msg)
    if ok and clear_ai:
        ss.ai_picks, ss.ai_idx = [], 0


def cb_add_from_search():
    res = ss.stop_results
    idx = ss.get(f"stop_radio_{ss.stop_search_no}", 0)
    if res and 0 <= idx < len(res):
        cb_add_stop(res[idx])
        ss.stop_results = []


def cb_move(i: int, d: int):
    j = i + d
    if 0 <= j < len(ss.stops):
        ss.stops[i], ss.stops[j] = ss.stops[j], ss.stops[i]


def cb_remove(i: int):
    if 0 <= i < len(ss.stops):
        name = ss.stops.pop(i)["name"]
        flash("stops", "info", f"Removed “{name}”.")


def cb_add_plan_selection():
    plan = ss.plan
    if not plan:
        return
    chosen = [p for i, p in enumerate(plan["picks"]) if ss.get(f"plan_{ss.plan_no}_{i}")]
    if not chosen:
        flash("stops", "warning", "Select at least one venue to add.")
        return
    added, blocked = 0, 0
    for p in chosen:
        ok, _ = add_stop(p)
        if ok:
            added += 1
        else:
            blocked += 1
    if blocked:
        flash("stops", "error",
              f"Added {added} of {len(chosen)} selected venues. The limit is {MAX_STOPS} "
              "stops (or a venue was already in your list).")
    else:
        flash("stops", "success", f"Added {added} venue(s). Your day is being built below.")
    if added:
        ss.auto_generate = True
    ss.plan = None


# --------------------------------------------------------------------------
# Start location (UC-01, BR-02)
# --------------------------------------------------------------------------
def set_start(place: dict, source: str):
    ss.start = {"id": place.get("id") or "start", "name": place["name"],
                "address": place.get("address", ""), "lat": place["lat"],
                "lng": place["lng"], "source": source}
    ss.start_results, ss.stop_results = [], []
    ss.ai_picks, ss.plan = [], None


def cb_clear_start():
    ss.start = None
    ss.gps_state = "idle"


def cb_gps_retry():
    ss.gps_state = "pending"


def cb_gps_accept():
    set_start(ss.gps_candidate, "gps")


def cb_gps_reject():
    ss.gps_candidate = None
    ss.gps_state = "idle"
    flash("start", "info", "OK, please pick a saved spot or search for an address.")


def cb_use_saved(name: str):
    try:
        if name in ss.saved:
            set_start(ss.saved[name], "saved")
        else:
            res = places_search(PRESET_SPOTS[name], n=1)
            if not res:
                flash("start", "error", "That saved spot could not be resolved. Try the search tab.")
                return
            set_start(dict(res[0], name=name.split(" (")[0] if "(" in name else res[0]["name"]), "saved")
    except ApiError as e:
        flash("start", "error", friendly(e))


def cb_set_start_from_results():
    res = ss.start_results
    idx = ss.get(f"start_radio_{ss.start_search_no}", 0)
    if res and 0 <= idx < len(res):
        set_start(res[idx], "search")


def cb_save_spot(label: str):
    s = ss.start
    if s:
        ss.saved[label] = {k: s[k] for k in ("id", "name", "address", "lat", "lng")}
        flash("start_saved", "success", f"Saved as “{label}” for this session.")


def gps_tab():
    state = ss.gps_state
    if state == "pending":
        st.caption("Your browser will ask for permission to use your location. "
                   "You can also use the Saved spots or Search tabs at any time.")
        loc = get_geolocation()
        if loc:
            coords = loc.get("coords") if isinstance(loc, dict) else None
            if coords and coords.get("latitude") is not None:
                lat, lng = coords["latitude"], coords["longitude"]
                try:
                    addr = reverse_geocode(lat, lng)
                except ApiError:
                    addr = None
                ss.gps_candidate = {
                    "id": "gps", "name": "Current location",
                    "address": addr or f"{lat:.5f}, {lng:.5f}", "lat": lat, "lng": lng}
                ss.gps_state = "found"
            else:
                ss.gps_state = "denied"   # BR-02: never crash, fall back
            st.rerun()
    elif state == "found":
        cand = ss.gps_candidate
        st.markdown(f"📍 **Detected location**  \n<span class='sub'>{safe(cand['address'])}</span>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("✅ Use this", on_click=cb_gps_accept, key="gps_ok")
        c2.button("✖ Not here", on_click=cb_gps_reject, key="gps_no")
    elif state == "denied":
        st.info("Location access is unavailable or was denied. No problem: use the "
                "**Saved spots** or **Search** tab to set your starting point.")
        st.button("Try location again", on_click=cb_gps_retry, key="gps_retry")
    else:
        st.button("📍 Use my current location", on_click=cb_gps_retry, key="gps_use")


def saved_tab():
    names = list(ss.saved) + list(PRESET_SPOTS)
    choice = st.selectbox("Choose a spot", names, key="saved_choice")
    st.button("Use this spot", on_click=cb_use_saved, args=(choice,), key="saved_use")
    st.caption("Home / Hotel can be saved for this session after you set a start point. "
               "Nothing is stored permanently.")


def start_search_tab():
    with st.form("start_form"):
        q = st.text_input("Address or place", placeholder="e.g. Konak Square, Izmir")
        go = st.form_submit_button("Search", use_container_width=True)
    if go:
        if not q.strip():
            st.warning("Please type an address or place name.")
        else:
            try:
                ss.start_results = places_search(q.strip(), n=5)
                ss.start_search_no += 1
                if not ss.start_results:
                    st.error("No match found. Check the spelling or add the city name.")
            except ApiError as e:
                st.error(friendly(e))
    if ss.start_results:
        res = ss.start_results
        st.radio("Select the correct match", range(len(res)), key=f"start_radio_{ss.start_search_no}",
                 format_func=lambda i: f"{res[i]['name']} - {res[i]['address']}")
        st.button("Set as starting point", on_click=cb_set_start_from_results, key="start_set")


def section_start():
    step_header(1, "Where are you starting?", "#10b981")
    if ss.start:
        s = ss.start
        addr = s["address"] if s["address"] and s["address"] != s["name"] else ""
        st.markdown(f"<div class='startcard'>📍 <b>{safe(s['name'])}</b>"
                    f"<div class='sub'>{safe(addr)}</div></div>", unsafe_allow_html=True)
        show_flash("start_saved")
        c1, c2, c3 = st.columns(3)
        c1.button("Change", on_click=cb_clear_start, key="chg_start")
        c2.button("Save: Home", on_click=cb_save_spot, args=("Home",), key="save_home")
        c3.button("Save: Hotel", on_click=cb_save_spot, args=("Hotel",), key="save_hotel")
        return
    show_flash("start")
    t1, t2, t3 = st.tabs(["📍 My location", "⭐ Saved", "🔎 Search"])
    with t1:
        gps_tab()
    with t2:
        saved_tab()
    with t3:
        start_search_tab()


# --------------------------------------------------------------------------
# Stops section: manual search (UC-02), AI pick (UC-03), daily plan (UC-06)
# --------------------------------------------------------------------------
def stars(p: dict) -> str:
    bits = []
    if p.get("rating"):
        bits.append(f"⭐ {p['rating']:.1f}" + (f" ({p['reviews']:,})" if p.get("reviews") else ""))
    if p.get("price"):
        bits.append(p["price"])
    if p.get("type"):
        bits.append(p["type"])
    return " · ".join(bits)


def pick_card(p: dict):
    meta = [stars(p), p["address"]]
    if p.get("slot") and p["slot"] != "any":
        meta.insert(1, f"🕐 {SLOTS[p['slot']]} · ~{p.get('visit_minutes', DEFAULT_STAY)} min")
    st.markdown(card_html("✨", p["name"], meta, "#7c3aed"), unsafe_allow_html=True)
    if p.get("rationale"):
        st.markdown(f"💡 {safe(p['rationale'])}", unsafe_allow_html=True)


def tab_search_stop(full: bool):
    anchor = ss.stops[-1] if ss.stops else ss.start
    with st.form("stop_form"):
        q = st.text_input("Venue name or address", placeholder="e.g. Kemeralti Bazaar")
        go = st.form_submit_button("Search", use_container_width=True, disabled=full)
    if go:
        if not q.strip():
            st.warning("Please type a venue name or address.")
        else:
            try:
                ss.stop_results = places_search(q.strip(), bias=anchor, n=5)
                ss.stop_search_no += 1
                if not ss.stop_results:
                    st.error("Location not found. Try another spelling.")
            except ApiError as e:
                st.error(friendly(e))
    if ss.stop_results and not full:
        res = ss.stop_results
        st.radio("Select the intended venue", range(len(res)), key=f"stop_radio_{ss.stop_search_no}",
                 format_func=lambda i: f"{res[i]['name']} - {res[i]['address']}")
        st.button("➕ Add to itinerary", on_click=cb_add_from_search, key="stop_add")


def tab_ai_pick(full: bool):
    cat = st.selectbox("Category", list(CATEGORIES), key="ai_cat")
    prefs = st.text_input("Preferences (optional)", key="ai_prefs",
                          placeholder="e.g. quiet, vegetarian, good for photos")
    if st.button("✨ Get AI popular pick", disabled=full, key="ai_go"):
        try:
            with st.spinner("Finding a popular, verified place..."):
                ss.ai_picks = ai_popular_picks(cat, prefs)
                ss.ai_idx = 0
            if not ss.ai_picks:
                st.warning("No results found. Try another category or search manually.")
        except ApiError as e:
            ss.ai_picks = []
            st.error("Recommendation unavailable right now. You can still add stops with the "
                     f"Search tab. ({e})")
    picks, idx = ss.ai_picks, ss.ai_idx
    if picks and not full:
        if idx < len(picks):
            with st.container(border=True):
                st.caption(f"Pick {idx + 1} of {len(picks)} · verified on Google Maps")
                pick_card(picks[idx])
                c1, c2 = st.columns(2)
                c1.button("➕ Add", on_click=cb_add_stop, args=(picks[idx], True), key=f"ai_add_{idx}")
                c2.button("🔄 Another", on_click=lambda: ss.update(ai_idx=ss.ai_idx + 1), key=f"ai_next_{idx}")
        else:
            st.info("No more picks for this search. Try another category or search manually.")


def tab_daily_plan(full: bool, remaining: int):
    st.caption("Tell me about your day. I pick the places, then build a timed agenda "
               "and the best route between them.")
    with st.form("plan_form"):
        interests = st.text_area("Interests", placeholder="e.g. history, street food, sea views",
                                 height=80)
        hours = st.slider("Available hours", 1, 12, 4)
        pace = st.select_slider("Pace", options=list(PACE_TARGET), value="Moderate")
        budget = st.selectbox("Budget", BUDGETS)
        go = st.form_submit_button("🗓️ Plan my day", use_container_width=True, disabled=full)
    if go:
        try:
            with st.spinner("Building your plan..."):
                ss.plan = daily_plan(interests, hours, pace, budget)
                ss.plan_no += 1
            if not ss.plan["picks"]:
                st.warning("No results found. Try different interests or search manually.")
                ss.plan = None
            else:
                ss.day_hours = hours
                ss.day_summary = ss.plan.get("summary", "")
        except ApiError as e:
            ss.plan = None
            st.error(f"Suggestions unavailable right now. You can still add stops manually. ({e})")
    plan = ss.plan
    if plan and not full:
        if plan.get("notice"):
            st.info(plan["notice"])
        if plan.get("summary"):
            st.markdown(f"<div class='daycard'>🗒️ {safe(plan['summary'])}</div>",
                        unsafe_allow_html=True)
        total = 0
        for i, p in enumerate(plan["picks"]):
            color = PALETTE[i % len(PALETTE)]
            on = st.checkbox(f"**{p['name']}**", value=i < remaining, key=f"plan_{ss.plan_no}_{i}")
            st.markdown(card_html(str(i + 1), p["name"],
                                  [stars(p), f"🕐 {SLOTS[p['slot']]} · ~{p['visit_minutes']} min",
                                   p["address"]], color)
                        + f"<div class='sub' style='margin:-.2rem 0 .6rem'>💡 {safe(p['rationale'])}</div>",
                        unsafe_allow_html=True)
            if on:
                total += p["visit_minutes"]
        st.caption(f"Selected visit time: ~{fmt_dur(total * 60)} of {plan['hours']} h available "
                   "(travel time not included).")
        st.button("➕ Add selected & build my day", on_click=cb_add_plan_selection,
                  key="plan_add", type="primary")


def section_stops():
    step_header(2, "Pick your stops", "#7c3aed")
    if not ss.start:
        st.info("Set a starting point first. Stops can be added once it is confirmed.")
        return
    n = len(ss.stops)
    full = n >= MAX_STOPS
    pips = "".join(f"<span class='pip{' on' if i < n else ''}' style='--c:{PALETTE[i]}'></span>"
                   for i in range(MAX_STOPS))
    st.markdown(f"<div class='pips'>{pips}<span class='cnt'>{n}/{MAX_STOPS} stops</span></div>",
                unsafe_allow_html=True)
    show_flash("stops")
    if full:
        st.warning(f"Limit reached: a maximum of {MAX_STOPS} stops is allowed. "
                   "Remove a stop to add another.")
    for i, s in enumerate(ss.stops):
        color = PALETTE[i % len(PALETTE)]
        st.markdown(card_html(str(i + 1), s["name"], [stars(s), s["address"]], color),
                    unsafe_allow_html=True)
        slot_keys = list(SLOTS)
        a, b = st.columns(2)
        s["visit_minutes"] = a.selectbox(
            "Stay", VISIT_OPTIONS, index=VISIT_OPTIONS.index(snap(s.get("visit_minutes"))),
            format_func=lambda m: f"⏱ {m} min", key=f"vm_{s['id']}")
        s["slot"] = b.selectbox(
            "Best time", slot_keys, index=slot_keys.index(s.get("slot", "any")),
            format_func=lambda k: f"🕐 {SLOTS[k]}", key=f"sl_{s['id']}")
        c1, c2, c3 = st.columns(3)
        c1.button("⬆", key=f"up_{s['id']}", on_click=cb_move, args=(i, -1),
                  disabled=i == 0, help="Move up")
        c2.button("⬇", key=f"dn_{s['id']}", on_click=cb_move, args=(i, 1),
                  disabled=i == n - 1, help="Move down")
        c3.button("🗑", key=f"rm_{s['id']}", on_click=cb_remove, args=(i,), help="Remove")
    t1, t2, t3 = st.tabs(["🔎 Search", "✨ AI pick", "🗓️ Daily plan"])
    with t1:
        tab_search_stop(full)
    with t2:
        tab_ai_pick(full)
    with t3:
        tab_daily_plan(full, MAX_STOPS - n)


# --------------------------------------------------------------------------
# Itinerary generation (UC-04), comparison (UC-05), export (UC-07)
# --------------------------------------------------------------------------
def current_signature() -> str:
    s = ss.start
    return json.dumps([
        (round(s["lat"], 5), round(s["lng"], 5)) if s else None,
        [p["id"] for p in ss.stops], bool(ss.optimize)])


def best_order(M: dict, n: int, visits: dict, slots: dict, start_min: int) -> list[int]:
    """Python sequencing logic (start fixed). Priorities: 1) avoid legs without a transit
    route, 2) respect each stop's best time of day (simulated clock), 3) minimize total
    public transit time."""
    best = None
    for perm in permutations(range(1, n + 1)):
        miss = tot = pen = 0
        prev, clock = 0, start_min
        for j in perm:
            v = M[(prev, j)]
            if v is None:
                miss += 1
                clock += 20
            else:
                tot += v
                clock += math.ceil(v / 60)
            pen += slot_penalty(slots[j], clock)
            clock += visits[j]
            prev = j
        if best is None or (miss, pen, tot) < best[0]:
            best = ((miss, pen, tot), perm)
    return list(best[1])


def generate():
    ss.gen_error = None
    if not ss.start:
        st.warning("Please set a starting point first.")
        return
    if not ss.stops:
        st.warning("Add at least one stop before generating the itinerary.")
        return
    t0 = time.perf_counter()
    start, stops = ss.start, list(ss.stops)
    n = len(stops)
    nodes = [start] + stops
    try:
        if ss.optimize and n > 1:
            pairs = [(i, j) for i in range(n + 1) for j in range(1, n + 1) if i != j]
            legs = ensure_legs([(nodes[i], nodes[j]) for i, j in pairs], "TRANSIT")
            M = {p: (l["seconds"] if l["status"] == "ok" else None) for p, l in zip(pairs, legs)}
            visits = {i + 1: int(stops[i].get("visit_minutes") or DEFAULT_STAY) for i in range(n)}
            slots = {i + 1: stops[i].get("slot", "any") for i in range(n)}
            dstart = ss.get("day_start_time", dtime(9, 0))
            order = best_order(M, n, visits, slots, dstart.hour * 60 + dstart.minute)
        else:
            order = list(range(1, n + 1))
        seq = [start] + [nodes[k] for k in order]
        pair_seq = list(zip(seq[:-1], seq[1:]))
        transit = ensure_legs(pair_seq, "TRANSIT")
    except ApiError as e:
        ss.gen_error = (friendly(e) + " Legs already retrieved stay cached, so only the "
                        "missing ones are requested next time. No partial total is shown.")
        ss.result = None
        return
    driving, driving_error = None, None
    try:
        driving = ensure_legs(pair_seq, "DRIVE")
    except ApiError as e:
        driving_error = str(e)

    legs_out = []
    for i, (a, b) in enumerate(pair_seq):
        legs_out.append({
            "from": a["name"], "to": b["name"], "transit": transit[i],
            "driving": driving[i] if driving else None,
            "url": maps_url(a, [b], "transit"),
        })
    t_ok = all(l["transit"]["status"] == "ok" for l in legs_out)
    d_ok = bool(driving) and all(l["driving"]["status"] == "ok" for l in legs_out)
    elapsed = time.perf_counter() - t0
    log.info("Itinerary generated: %d stops, %.2fs", n, elapsed)
    ss.result = {
        "signature": current_signature(),
        "sequence": [p["name"] for p in seq],
        "ordered": seq,
        "legs": legs_out,
        "transit_total": sum(l["transit"]["seconds"] for l in legs_out) if t_ok else None,
        "driving_total": sum(l["driving"]["seconds"] for l in legs_out) if d_ok else None,
        "driving_error": driving_error,
        "export_url": maps_url(start, seq[1:], "driving"),
        "elapsed": elapsed,
    }


def cmp_html(t_total, d_total) -> str:
    faster = None
    if t_total is not None and d_total is not None and t_total != d_total:
        faster = "transit" if t_total < d_total else "driving"

    def box(icon, label, val, key):
        win = faster == key
        txt = fmt_dur(val) if val is not None else "unavailable"
        badge = "<span class='badge'>Faster</span>" if win else ""
        return (f"<div class='box{' win' if win else ''}'><div class='lbl'>{icon} {label}</div>"
                f"<div class='val'>{txt}</div>{badge}</div>")

    return ("<div class='cmp'>" + box("🚌", "Public transit", t_total, "transit")
            + box("🚗", "Driving", d_total, "driving") + "</div>")


def fare_summary(legs: list[dict]):
    totals: dict[str, float] = {}
    missing = []
    for i, l in enumerate(legs, 1):
        t = l["transit"]
        f = t.get("fare")
        if f:
            totals[f["currency"]] = totals.get(f["currency"], 0) + f["amount"]
        elif not (t["status"] == "ok" and t.get("seconds") == 0):
            missing.append(i)
    if not totals:
        return "fare not available", None
    head = " + ".join(fmt_money(v, c) for c, v in totals.items())
    note = (f"incomplete: fare not available for leg(s) {', '.join(map(str, missing))}"
            if missing else None)
    return head, note


def build_agenda(r: dict, day_start: datetime, mode: str):
    """Turn the optimized route into a timed agenda (travel + stay for every stop)."""
    seq, legs = r["ordered"], r["legs"]
    by_id = {s["id"]: s for s in ss.stops}
    t = day_start
    items = [{"kind": "start", "start": t, "end": t, "title": seq[0]["name"],
              "sub": "Your day begins", "color": "#10b981"}]
    unknown, mismatch = [], []
    for i, leg in enumerate(legs):
        dest = seq[i + 1]
        stop = by_id.get(dest["id"], dest)
        tr, dr = leg["transit"], leg["driving"]
        order = [("🚌", tr), ("🚗", dr)] if mode == "Public transit" else [("🚗", dr), ("🚌", tr)]
        chosen = next(((ic, x) for ic, x in order if x and x["status"] == "ok"), None)
        secs, icon, note = 0, "❔", ""
        if chosen:
            icon, x = chosen
            secs = x["seconds"]
            if (icon == "🚌") != (mode == "Public transit"):
                note = "no route for the chosen mode, other mode used"
            elif icon == "🚌" and x.get("fare"):
                note = "est. fare " + fmt_money(x["fare"]["amount"], x["fare"]["currency"])
        else:
            unknown.append(i + 1)
        depart = t
        t = t + timedelta(minutes=math.ceil(secs / 60))
        items.append({"kind": "travel", "start": depart, "end": t,
                      "title": f"{icon} {fmt_dur(secs) if chosen else 'travel time unknown'} to {dest['name']}",
                      "sub": note})
        stay = int(stop.get("visit_minutes") or DEFAULT_STAY)
        arrive = t
        t = t + timedelta(minutes=stay)
        slot = stop.get("slot", "any")
        if slot_penalty(slot, arrive.hour * 60 + arrive.minute) > 0:
            mismatch.append(f"{dest['name']} starts at {arrive:%H:%M}, outside its preferred "
                            f"{SLOTS[slot].lower()} window.")
        items.append({"kind": "visit", "start": arrive, "end": t, "title": dest["name"],
                      "sub": f"{stay} min · until {t:%H:%M}", "addr": dest.get("address", ""),
                      "extra": stop.get("rationale") or "", "color": PALETTE[i % len(PALETTE)]})
    items.append({"kind": "end", "start": t, "end": t, "title": "Day wraps up 🎉",
                  "sub": "", "color": "#f59e0b"})
    return items, unknown, mismatch


def timeline_html(items: list[dict]) -> str:
    rows = []
    for k, it in enumerate(items):
        color = it.get("color", "#94a3b8")
        last = k == len(items) - 1
        if it["kind"] == "travel":
            dot = "<span class='tl-dot sm'></span>"
            body = (f"<div class='tl-travel'>{safe(it['title'])}"
                    + (f"<div class='sub'>{safe(it['sub'])}</div>" if it["sub"] else "") + "</div>")
        elif it["kind"] == "visit":
            dot = "<span class='tl-dot'></span>"
            extra = f"<div class='sub'>💡 {safe(it['extra'])}</div>" if it["extra"] else ""
            body = (f"<div class='tl-card'><div class='ttl'>{safe(it['title'])}</div>"
                    f"<div class='sub'>{safe(it['sub'])}</div>"
                    f"<div class='sub'>{safe(it['addr'])}</div>{extra}</div>")
        elif it["kind"] == "start":
            dot = "<span class='tl-dot'></span>"
            body = (f"<div class='tl-card'><div class='ttl'>🟢 {safe(it['title'])}</div>"
                    f"<div class='sub'>{safe(it['sub'])}</div></div>")
        else:
            dot = "<span class='tl-dot'></span>"
            body = f"<div class='tl-end'>{safe(it['title'])}</div>"
        line = "" if last else "<span class='tl-line'></span>"
        rows.append(f"<div class='tl-row' style='--c:{color}'>"
                    f"<div class='tl-time'>{it['start']:%H:%M}</div>"
                    f"<div class='tl-rail'>{dot}{line}</div><div class='tl-body'>{body}</div></div>")
    return "<div class='tl'>" + "".join(rows) + "</div>"


def ics_escape(text) -> str:
    return (str(text or "").replace("\\", "\\\\").replace(";", "\\;")
            .replace(",", "\\,").replace("\n", "\\n"))


def build_ics(items: list[dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//RouteGuide//Daily Plan//EN",
           "CALSCALE:GREGORIAN"]
    for n, it in enumerate(items):
        if it["kind"] not in ("visit", "travel"):
            continue
        title = it["title"] if it["kind"] == "visit" else "Travel: " + it["title"]
        out += ["BEGIN:VEVENT", f"UID:rg-{stamp}-{n}@routeguide", f"DTSTAMP:{stamp}",
                f"DTSTART:{it['start']:%Y%m%dT%H%M%S}", f"DTEND:{it['end']:%Y%m%dT%H%M%S}",
                f"SUMMARY:{ics_escape(title)}"]
        if it.get("addr"):
            out.append(f"LOCATION:{ics_escape(it['addr'])}")
        desc = it.get("extra") or it.get("sub")
        if desc:
            out.append(f"DESCRIPTION:{ics_escape(desc)}")
        out.append("END:VEVENT")
    out.append("END:VCALENDAR")
    return "\r\n".join(out)


def render_agenda(r: dict):
    mode = st.radio("Travel between stops by", ["Public transit", "Driving"],
                    horizontal=True, key="agenda_mode")
    day_start = datetime.combine(ss.get("day_date", date.today()),
                                 ss.get("day_start_time", dtime(9, 0)))
    items, unknown, mismatch = build_agenda(r, day_start, mode)
    if ss.day_summary:
        st.markdown(f"<div class='daycard'>🗒️ {safe(ss.day_summary)}</div>", unsafe_allow_html=True)
    st.markdown(timeline_html(items), unsafe_allow_html=True)
    total_min = int((items[-1]["start"] - day_start).total_seconds() // 60)
    st.caption(f"Your day runs {day_start:%H:%M} → {items[-1]['start']:%H:%M} "
               f"({fmt_dur(total_min * 60)} including travel).")
    if ss.day_hours:
        if total_min > ss.day_hours * 60:
            st.warning(f"This plan takes about {fmt_dur(total_min * 60)}, more than your "
                       f"{ss.day_hours} h. Remove a stop or shorten a stay.")
        else:
            st.success(f"Fits your {ss.day_hours} h. 👍")
    if unknown:
        st.warning(f"Travel time is unknown for leg(s) {', '.join(map(str, unknown))}, "
                   "so those legs are counted as 0 min.")
    for m in mismatch[:3]:
        st.info("⏰ " + m)
    if mode == "Public transit":
        st.caption("ℹ️ " + FARE_DISCLAIMER)
    st.caption("Changed the start time, stay times or best times? Press **Generate** again "
               "to re-sequence the stops (cached legs are reused).")
    st.download_button("📅 Add to calendar (.ics)", data=build_ics(items),
                       file_name="routeguide-day-plan.ics", mime="text/calendar", key="ics_dl")


def render_route(r: dict):
    legs = r["legs"]
    st.markdown("#### Optimized route")
    st.markdown("  \n".join(
        ("🟢 " if i == 0 else f"{i}️⃣ " if i < 10 else f"{i}. ") + safe(name)
        for i, name in enumerate(r["sequence"])), unsafe_allow_html=True)

    st.markdown("#### Total travel time")
    st.markdown(cmp_html(r["transit_total"], r["driving_total"]), unsafe_allow_html=True)
    t, d = r["transit_total"], r["driving_total"]
    if t is not None and d is not None and t != d:
        st.caption(f"{'Public transit' if t < d else 'Driving'} is {fmt_dur(abs(t - d))} faster "
                   "for the same stop order. Driving cost is not calculated.")
    if t is None:
        bad = [str(i) for i, l in enumerate(legs, 1) if l["transit"]["status"] != "ok"]
        st.warning(f"Total transit duration is unavailable: no transit route found for leg(s) "
                   f"{', '.join(bad)}. Replace or remove that stop.")
    if d is None:
        st.warning("Driving duration is unavailable right now"
                   + (f" ({r['driving_error']})" if r["driving_error"] else "")
                   + ". Your transit results are kept.")

    head, note = fare_summary(legs)
    st.markdown("<div class='fare'>Estimated public transit fare<br><b>" + safe(head) + "</b>"
                + (f"<br><span class='sub'>{safe(note)}</span>" if note else "") + "</div>",
                unsafe_allow_html=True)
    st.caption("ℹ️ " + FARE_DISCLAIMER)

    st.markdown("#### Leg by leg")
    for i, l in enumerate(legs, 1):
        t_, d_ = l["transit"], l["driving"]
        if t_["status"] == "ok":
            tx = f"🚌 {fmt_dur(t_['seconds'])}"
            tx += (" · " + fmt_money(t_["fare"]["amount"], t_["fare"]["currency"])
                   if t_.get("fare") else " · fare not available")
        else:
            tx = "🚌 no transit route found · fare not available"
        dx = ("🚗 " + (fmt_dur(d_["seconds"]) if d_ and d_["status"] == "ok" else "unavailable"))
        with st.container(border=True):
            st.markdown(f"**{i}. {safe(l['from'])} → {safe(l['to'])}**  \n{tx}  \n{dx}  \n"
                        f"[Open this leg in Google Maps (transit)]({l['url']})",
                        unsafe_allow_html=True)

    st.markdown("#### Export")
    st.link_button("🗺️ Open in Google Maps", r["export_url"], type="primary",
                   use_container_width=True)
    st.caption("Google Maps does not support intermediate stops in transit mode, so the full "
               "route opens in driving mode with every stop in the optimized order. Use the "
               "per-leg links above for transit directions.")
    with st.expander("Copy link"):
        st.code(r["export_url"], language=None)
    st.caption(f"Calculated in {r['elapsed']:.1f} s (target: under 3 s; cached legs are reused).")
    st.button("🔄 Refresh live data", on_click=lambda: ss.cache.pop("legs", None),
              key="refresh_legs", help="Clears cached legs so the next calculation is fresh.")


def render_results():
    r = ss.result
    if ss.gen_error:
        st.error(ss.gen_error)
    if not r:
        return
    if r["signature"] != current_signature():
        st.info("Your start point or stops changed after the last calculation. "
                "The old agenda, route and export link are disabled: press "
                "**Build my day** again.")
        return
    t1, t2 = st.tabs(["🗓️ Daily agenda", "🗺️ Route & compare"])
    with t1:
        render_agenda(r)
    with t2:
        render_route(r)


def section_generate():
    step_header(3, "Your day & route", "#f59e0b")
    if not ss.start:
        st.info("Set a starting point and add stops to build your day.")
        return
    c1, c2 = st.columns(2)
    c1.date_input("Date", value=date.today(), key="day_date")
    c2.time_input("Day starts at", value=dtime(9, 0), key="day_start_time", step=900)
    st.checkbox("Optimize stop order (shortest transit, right time of day)", key="optimize")
    go = st.button("🚀 Build my day", type="primary", use_container_width=True, key="gen")
    if go or ss.pop("auto_generate", False):
        with st.spinner("Calculating legs and building your agenda..."):
            generate()
    render_results()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)
    if not MAPS_KEY:
        st.error("GOOGLE_MAPS_API_KEY is missing. Add it to Streamlit secrets "
                 "(see README.md) and reload.")
        st.stop()
    init_state()
    if not GEMINI_KEY:
        st.caption("⚠️ GEMINI_API_KEY not set: AI recommendations are unavailable, "
                   "manual search and routing still work.")
    section_start()
    st.divider()
    section_stops()
    st.divider()
    section_generate()
    st.divider()
    st.caption("Stateless session: no login, nothing is stored after you close or reset the app.")
    st.button("Reset session", on_click=lambda: ss.clear(), key="reset")


main()
