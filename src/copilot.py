"""Situation-report copilot: deterministic tools -> (optional LLM wording) -> numeric verification -> fallback.

Guarantee ("how do you know the LLM didn't invent that number?"):
  1. Every number comes from `src.facts` tools reading data/outputs/*; the model is only asked to word them.
  2. Whatever the model writes is scanned: every number in the text must appear in (or be a faithful rounding of)
     the tool JSON it was given. One unverifiable number rejects the whole answer.
  3. A rejected or failed LLM answer is replaced by a template answer that is deterministic by construction.
  With no LLM configured (default) the template answer is used directly. Nepali templates are included so the
  Nepali report never depends on machine translation altering a number; they need a native-speaker review.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

import requests

from .facts import Facts

DEVANAGARI = str.maketrans("०१२३४५६७८९", "0123456789")
LABELS = {"en": "English", "ne": "Nepali"}


# ============================================================================= verification
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_IDENT = re.compile(r"(?:Sentinel|सेन्टिनल)-\d(?:/\d)?|[A-Za-z]+[-_]?\d+(?:\.\d+)?")  # Sentinel-1/2, S2, EMSR927, km2; NOT "about 37" (space)
_LISTMARK = re.compile(r"(?m)^\s*(?:[-*•]\s*)?\d+[.)]\s+")


def numbers_in(text: str) -> list[float]:
    t = _LISTMARK.sub("", _IDENT.sub(" ", str(text).translate(DEVANAGARI)))
    out = []
    for m in _NUM.findall(t):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def allowed_numbers(data) -> set[float]:
    s: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            s.add(float(x))
            if 0 < abs(x) <= 1:
                s.add(float(x) * 100)
        elif isinstance(x, str):
            s.update(numbers_in(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            s.add(float(len(x)))
            for v in x:
                walk(v)

    walk(data)
    return s


def _decimals(n: float, text_hint: float) -> int:
    t = repr(round(text_hint, 6))
    return len(t.split(".")[1].rstrip("0")) if "." in t else 0


def verify_numbers(text: str, data) -> tuple[bool, list[float]]:
    """True if every number in `text` is present in `data` or a faithful rounding of one of its numbers."""
    allowed = allowed_numbers(data)
    bad = []
    for n in numbers_in(text):
        d = _decimals(n, n)
        tol = 0.5 * 10 ** (-d) + 1e-9
        if not any(abs(a - n) <= tol for a in allowed):
            bad.append(n)
    return (not bad), bad


# ============================================================================= templates
def _n(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _names(items) -> str:
    return ", ".join(f"{i['name']}, buildings nearby {i['buildings_300m']}" for i in items)


def _names_ne(items) -> str:
    return ", ".join(f"{i['name']} (नजिक {i['buildings_300m']} भवन)" for i in items)


def render_en(intent: str, d: dict) -> str:
    syn = "SYNTHETIC DEMO DATA, not a real event. " if _flag(d, "synthetic") else ""
    if intent == "summary":
        e, dm, c = d["extent"], d["damage"], d["cutoff"]
        t = (f"{syn}Situation report for {e['event']} ({e['date']}). Sentinel-1/2 mapping shows {_n(e['flood_area_km2'])} km2 of "
             f"new flood water and {_n(e['debris_area_km2'])} km2 of debris or scour. Against the pre-event OpenStreetMap "
             f"snapshot: buildings inside the hazard zone {dm['buildings_hit']} ({dm['buildings_high_overlap']} with high "
             f"overlap); road affected {_n(dm['road_km_in_hazard'])} km; mapped bridges exposed {dm['bridges_exposed']} of "
             f"{dm['bridges_total']}. ")
        t += (f"Settlements cut off from hospital or town access: {c['cut_off_count']}"
              + (f" ({_names(c['cut_off'])})" if c["cut_off"] else "")
              + f". Additional settlements cut off only if the exposed bridges failed: {c['cut_off_if_bridges_fail_count']}"
              + (f" ({_names(c['cut_off_if_bridges_fail'])})" if c["cut_off_if_bridges_fail"] else "") + ". ")
        t += (f"Radar could not observe {e['radar_unobservable_pct']}% of the area and OSM is incomplete, so figures "
              "are lower bounds. This is an automated estimate and needs human review.")
        return t
    if intent == "cutoff":
        t = f"{syn}Settlements cut off: {d['cut_off_count']}"
        t += f" ({_names(d['cut_off'])}). " if d["cut_off"] else ". "
        t += f"Additional settlements cut off only if exposed bridges failed: {d['cut_off_if_bridges_fail_count']}"
        t += f" ({_names(d['cut_off_if_bridges_fail'])}). " if d["cut_off_if_bridges_fail"] else ". "
        t += (f"Settlements analysed: {d['settlements_analysed']}; without a mapped road route in OpenStreetMap "
              f"(not counted as cut off): {d['settlements_without_mapped_route']}. Road access only; walking paths excluded.")
        return t
    if intent == "buildings":
        return (f"{syn}Buildings inside the hazard zone: {d['buildings_hit']} (high overlap {d['buildings_high_overlap']}, "
                f"moderate {d['buildings_moderate_overlap']}, low {d['buildings_low_overlap']}). Footprints come from "
                "pre-event OpenStreetMap and are lower bounds; overlap is not a damage grade.")
    if intent == "roads":
        return (f"{syn}Mapped road inside the flood or debris footprint: {_n(d['road_km_in_hazard'])} km "
                "(pre-event OpenStreetMap, motorable and track classes).")
    if intent == "bridges":
        names = f" ({', '.join(d['bridges_exposed_names'])})" if d["bridges_exposed_names"] else ""
        return (f"{syn}Mapped bridges exposed to new flood or debris: {d['bridges_exposed']} of {d['bridges_total']}{names}. "
                "Exposed means inside the hazard footprint, not collapsed.")
    if intent == "extent":
        return (f"{syn}New flood water covers {_n(d['flood_area_km2'])} km2 and debris or scour {_n(d['debris_area_km2'])} km2. "
                f"Radar observed {d['radar_valid_pct']}% of the area and could not observe {d['radar_unobservable_pct']}% "
                "(layover or shadow), so extent is a lower bound.")
    if intent == "flowpath":
        if not d.get("available"):
            return "No flow-path analysis has been run for this event."
        s = ", ".join(f"{x['name']} ({_n(x['km_downstream'])} km downstream)" for x in d["settlements"])
        return (f"{syn}The traced flow path is {_n(d['length_km'])} km long; settlements within its corridor: "
                f"{d['settlement_count']}" + (f" ({s})" if s else "") + ". It is a likely corridor, not an inundation model.")
    if intent == "limitations":
        return "Known limitations: " + " ".join(d["items"])
    if intent == "out_of_scope":
        return ("The system does not estimate that (for example casualties, population, rainfall, forecasts or costs). "
                "It can report flood and debris extent, buildings and roads in the hazard zone, bridge exposure, "
                "cut-off settlements, the flow path and known limitations.")
    raise ValueError(intent)


def render_ne(intent: str, d: dict) -> str:
    syn = "कृत्रिम डेमो डेटा, वास्तविक घटना होइन। " if _flag(d, "synthetic") else ""
    if intent == "summary":
        e, dm, c = d["extent"], d["damage"], d["cutoff"]
        t = (f"{syn}स्थिति प्रतिवेदन: {e['event']} ({e['date']})। सेन्टिनल-१/२ नक्सांकनअनुसार {_n(e['flood_area_km2'])} वर्ग किमी "
             f"नयाँ बाढी क्षेत्र र {_n(e['debris_area_km2'])} वर्ग किमी भग्नावशेष वा कटान क्षेत्र देखिएको छ। बाढी अघिको "
             f"OpenStreetMap अनुसार {dm['buildings_hit']} वटा भवन प्रभावित क्षेत्रभित्र पर्छन् (तीमध्ये {dm['buildings_high_overlap']} "
             f"वटामा उच्च अतिव्यापन), {_n(dm['road_km_in_hazard'])} किमी सडक प्रभावित छ, र नक्सांकित {dm['bridges_total']} पुलमध्ये "
             f"{dm['bridges_exposed']} जोखिममा छन्। ")
        t += (f"{c['cut_off_count']} वटा बस्ती अस्पताल वा सहरसँगको सडक सम्पर्कबाट विच्छेद भएका छन्"
              + (f" ({_names_ne(c['cut_off'])})" if c["cut_off"] else "")
              + f"; जोखिममा परेका पुल भत्किएमा थप {c['cut_off_if_bridges_fail_count']} वटा विच्छेद हुनेछन्"
              + (f" ({_names_ne(c['cut_off_if_bridges_fail'])})" if c["cut_off_if_bridges_fail"] else "") + "। ")
        t += (f"रडारले {e['radar_unobservable_pct']}% क्षेत्र हेर्न सकेन र OSM अपूर्ण हुन सक्छ, त्यसैले सङ्ख्याहरू न्यूनतम अनुमान हुन्। "
              "यो स्वचालित अनुमान हो; मानवीय पुष्टि आवश्यक छ।")
        return t
    if intent == "cutoff":
        t = f"{syn}{d['cut_off_count']} वटा बस्ती विच्छेद भएका छन्"
        t += f": {_names_ne(d['cut_off'])}। " if d["cut_off"] else "। "
        t += f"जोखिममा परेका पुल भत्किएमा मात्र थप {d['cut_off_if_bridges_fail_count']} वटा विच्छेद हुन्छन्"
        t += f": {_names_ne(d['cut_off_if_bridges_fail'])}। " if d["cut_off_if_bridges_fail"] else "। "
        t += (f"विश्लेषण गरिएका {d['settlements_analysed']} बस्तीमध्ये {d['settlements_without_mapped_route']} वटाको "
              "OpenStreetMap मा सडक मार्ग नक्सांकित छैन, ती विच्छेद गणनामा परेका छैनन्। केवल सडक पहुँच; पैदलमार्ग समावेश छैन।")
        return t
    if intent == "buildings":
        return (f"{syn}{d['buildings_hit']} वटा भवन प्रभावित क्षेत्रभित्र छन्: उच्च अतिव्यापन {d['buildings_high_overlap']}, मध्यम "
                f"{d['buildings_moderate_overlap']}, न्यून {d['buildings_low_overlap']}। भवनको आकार बाढी अघिको OpenStreetMap "
                "बाट हो र यो न्यूनतम सङ्ख्या हो; अतिव्यापन क्षतिको स्तर होइन।")
    if intent == "roads":
        return f"{syn}{_n(d['road_km_in_hazard'])} किमी नक्सांकित सडक बाढी वा भग्नावशेष क्षेत्रभित्र पर्छ।"
    if intent == "bridges":
        return (f"{syn}नक्सांकित {d['bridges_total']} पुलमध्ये {d['bridges_exposed']} वटा नयाँ बाढी वा भग्नावशेषको जोखिममा छन्। "
                "जोखिममा भनेको प्रभावित क्षेत्रभित्र हुनु हो, भत्किनु होइन।")
    if intent == "extent":
        return (f"{syn}नयाँ बाढी क्षेत्र {_n(d['flood_area_km2'])} वर्ग किमी र भग्नावशेष वा कटान {_n(d['debris_area_km2'])} वर्ग किमी छ। "
                f"रडारले {d['radar_valid_pct']}% क्षेत्र हेर्‍यो र {d['radar_unobservable_pct']}% हेर्न सकेन, त्यसैले यो न्यूनतम अनुमान हो।")
    if intent == "flowpath":
        if not d.get("available"):
            return "यस घटनाका लागि बहाव मार्ग विश्लेषण गरिएको छैन।"
        s = ", ".join(f"{x['name']} ({_n(x['km_downstream'])} किमी)" for x in d["settlements"])
        return (f"{syn}पत्ता लगाइएको बहाव मार्ग {_n(d['length_km'])} किमी लामो छ र {d['settlement_count']} वटा बस्ती नजिकबाट जान्छ"
                + (f": {s}" if s else "") + "। यो सम्भावित मार्ग हो, डुबान मोडेल होइन।")
    if intent == "limitations":
        return ("सीमाहरू: सेन्टिनल-१ ले ६ देखि १२ दिनमा दोहोरिन्छ त्यसैले यो घटनापछिको नक्सा हो, पूर्वचेतावनी होइन; भिरालो भूभागमा "
                "रडार छायाँले केही बाढी छुट्छ; मनसुनमा बादलले सेन्टिनल-२ प्रायः प्रयोग हुँदैन; OSM अपूर्ण छ; क्षति वा पुल "
                "भत्किएको पुष्टि यी नक्सांबाट हुँदैन।")
    if intent == "out_of_scope":
        return ("प्रणालीले यो अनुमान गर्दैन (जस्तै: मृतक, जनसङ्ख्या, वर्षा, पूर्वानुमान, लागत)। यसले बाढी र भग्नावशेषको क्षेत्र, प्रभावित "
                "भवन र सडक, पुलको जोखिम, विच्छेद भएका बस्ती, बहाव मार्ग र सीमाहरू बताउन सक्छ।")
    raise ValueError(intent)


def _flag(d, key) -> bool:
    if isinstance(d, dict):
        if d.get(key):
            return True
        return any(_flag(v, key) for v in d.values() if isinstance(v, dict))
    return False


# ============================================================================= routing
OUT_OF_SCOPE = ("casualt", "death", "dead", "died", "killed", "injur", "missing people", "how many people", "population",
                "rainfall", "forecast", "predict", "will it", "cost", "rupee", "evacuat", "मृत", "घाइते", "जनसंख्या")
ROUTES = [
    ("cutoff", ("cut off", "cut-off", "cutoff", "isolat", "village", "settlement", "hospital", "stranded", "access",
                "बस्ती", "गाउँ", "विच्छेद", "कटअफ", "अस्पताल")),
    ("bridges", ("bridge", "पुल")),
    ("roads", ("road", "highway", "सडक")),
    ("buildings", ("building", "house", "भवन", "घर")),
    ("flowpath", ("flow path", "flowpath", "downstream", "along the river", "बहाव")),
    ("limitations", ("limit", "caveat", "reliab", "trust", "accura", "सीमा")),
    ("extent", ("extent", "how large", "how big", "flooded area", "area", "km2", "क्षेत्र")),
    ("summary", ("summary", "situation", "report", "overview", "what happened", "स्थिति")),
]


def route(question: str) -> str:
    q = question.lower()
    if any(k in q for k in OUT_OF_SCOPE):
        return "out_of_scope"
    for intent, keys in ROUTES:
        if any(k in q for k in keys):
            return intent
    return "summary"


def tool_for(intent: str, facts: Facts) -> dict:
    return {"summary": facts.overview, "cutoff": facts.list_cutoff_settlements, "buildings": facts.count_flooded_buildings,
            "roads": facts.summarize_damage, "bridges": facts.summarize_damage, "extent": facts.flood_extent,
            "flowpath": facts.settlements_along_flowpath, "limitations": facts.limitations,
            "out_of_scope": lambda: {"sources": []}}[intent]()


# ============================================================================= LLM backends (optional)
class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, system: str, user: str) -> str:
        r = requests.post(f"{self.host}/api/generate", timeout=120, json={
            "model": self.model, "system": system, "prompt": user, "stream": False, "options": {"temperature": 0}})
        r.raise_for_status()
        return r.json()["response"].strip()


class OpenAILLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.key = os.environ["OPENAI_API_KEY"]

    def generate(self, system: str, user: str) -> str:
        r = requests.post("https://api.openai.com/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "temperature": 0,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def get_llm() -> LLM | None:
    kind = os.getenv("COPILOT_LLM", "none").lower()
    return {"ollama": OllamaLLM, "openai": OpenAILLM}.get(kind, lambda: None)()


SYSTEM = ("You are a disaster response assistant. Use ONLY the JSON provided by the user message. Copy numbers exactly; "
          "never add, compute, estimate or round numbers, and never invent places. If the JSON does not contain what "
          "is asked, say the system does not have it. Write in {lang}, at most 200 words, plain text.")


# ============================================================================= public API
def answer(question: str, lang: str, facts: Facts, llm: LLM | None = None) -> dict:
    lang = lang if lang in LABELS else "en"
    intent = route(question)
    data = tool_for(intent, facts)
    data = data | {"synthetic": facts.event["synthetic"]} if isinstance(data, dict) else data
    template = (render_ne if lang == "ne" else render_en)(intent, data)
    res = {"answer": template, "intent": intent, "lang": lang, "sources": data.get("sources", []),
           "mode": "template", "verified": True, "data": data}
    if llm is not None and intent != "out_of_scope":
        try:
            text = llm.generate(SYSTEM.format(lang=LABELS[lang]),
                                f"Question: {question}\nJSON:\n{json.dumps(data, ensure_ascii=False)}")
            ok, bad = verify_numbers(text, data)
            if ok and text:
                res.update(answer=text, mode="llm")
            else:
                res["llm_rejected"] = {"unverified_numbers": bad}
        except Exception as e:  # network down, model missing, ...
            res["llm_error"] = str(e)[:200]
    return res


def situation_report(facts: Facts, lang: str = "en", llm: LLM | None = None) -> dict:
    return answer("situation summary report", lang, facts, llm)
