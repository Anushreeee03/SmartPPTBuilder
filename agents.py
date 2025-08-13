# agents.py
import os
import re
import json
import html
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from readability import Document
from duckduckgo_search import DDGS

# optional better extractor
try:
    import trafilatura
    HAS_TRAFILATURA = True
except Exception:
    HAS_TRAFILATURA = False

# embeddings & ML
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# embedding model (fast & good)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- SEARCH ----------------
def search_duckduckgo(query, max_results=6):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title") or r.get("body") or "Untitled"
                url = r.get("href")
                if url:
                    results.append({"title": title.strip(), "url": url.strip()})
    except Exception:
        pass
    return results

# ---------------- EXTRACTION ----------------
def extract_article_text(url, max_chars=20000):
    """
    Return {"title","text","domain","url"}.
    Prefer trafilatura if available, else readability + BS4 fallback.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        html_text = resp.text

        text = ""
        title = ""
        if HAS_TRAFILATURA:
            try:
                downloaded = trafilatura.extract(html_text, include_comments=False, include_tables=False)
                if downloaded:
                    text = re.sub(r"\s+", " ", downloaded).strip()
            except Exception:
                text = ""
        if not text:
            # readability fallback
            doc = Document(html_text)
            title = doc.short_title() or ""
            body_html = doc.summary()
            soup = BeautifulSoup(body_html, "html.parser")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join([p for p in paragraphs if p])
            text = re.sub(r"\s+", " ", html.unescape(text)).strip()

        if not title:
            # try to get from page head
            soup_h = BeautifulSoup(html_text, "html.parser")
            ttag = soup_h.find("title")
            title = ttag.get_text(" ", strip=True) if ttag else url

        domain = urlparse(url).netloc
        if len(text) > max_chars:
            text = text[:max_chars]
        return {"title": title, "text": text, "domain": domain, "url": url}
    except Exception:
        return {"title": url, "text": "", "domain": "", "url": url}

# ---------------- SENTENCE OPS ----------------
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def split_sentences(text, min_words=6):
    if not text:
        return []
    # normalize whitespace and remove stray newlines inside sentences
    text = re.sub(r'\s+', ' ', text).strip()
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.split()) >= min_words]
    # filter out obviously noisy short fragments
    sents = [s for s in sents if len(s) > 30]
    return sents

def dedupe_sentences(sents, threshold=0.82, max_keep=400):
    if not sents:
        return []
    try:
        embs = EMBED_MODEL.encode(sents, show_progress_bar=False)
    except Exception:
        return sents[:max_keep]
    keep = []
    used = set()
    for i, emb in enumerate(embs):
        if i in used:
            continue
        keep.append(sents[i])
        sims = cosine_similarity([emb], embs)[0]
        similar_idxs = [j for j, sim in enumerate(sims) if sim >= threshold]
        for j in similar_idxs:
            used.add(j)
        if len(keep) >= max_keep:
            break
    return keep

# ---------------- TOPIC CLASSIFICATION ----------------
# fixed topics required by user
TOPIC_NAMES = ["Introduction", "Objective", "Features", "Importance", "Limitations", "Conclusion"]

# keywords to seed classification (keeps things reliable)
TOPIC_KEYWORDS = {
    "Introduction": ["what is", "introduction", "overview", "about", "definition", "meaning"],
    "Objective": ["objective", "goal", "purpose", "aim", "aims", "intent"],
    "Features": ["feature", "capability", "characteristic", "property", "supports", "functional"],
    "Importance": ["importance", "benefit", "advantage", "value", "useful", "relevance"],
    "Limitations": ["limit", "limitation", "challenge", "issue", "drawback", "risk", "concern"],
    "Conclusion": ["conclusion", "summary", "in summary", "takeaway", "future", "future work"]
}

def classify_sentences_into_topics(sents):
    """
    Hybrid classification:
     - keyword match boosts score
     - semantic similarity to topic prototype embeddings also used
    Returns dict topic -> list of (sentence, score)
    """
    if not sents:
        return {t: [] for t in TOPIC_NAMES}

    # compute embeddings for sentences
    try:
        sent_embs = EMBED_MODEL.encode(sents, show_progress_bar=False)
    except Exception:
        sent_embs = None

    # build topic prototypes by encoding keywords
    topic_proto_emb = {}
    for t in TOPIC_NAMES:
        kws = TOPIC_KEYWORDS.get(t, [])
        proto_text = " ".join(kws) if kws else t
        try:
            topic_proto_emb[t] = EMBED_MODEL.encode([proto_text])[0]
        except Exception:
            topic_proto_emb[t] = None

    results = {t: [] for t in TOPIC_NAMES}
    for idx, s in enumerate(sents):
        s_low = s.lower()
        # keyword score (count of keywords)
        kw_score = 0
        for t, kws in TOPIC_KEYWORDS.items():
            for kw in kws:
                if kw in s_low:
                    kw_score += 1
        # semantic score: cosine with prototype (if available)
        sem_scores = {}
        if sent_embs is not None:
            for t, proto in topic_proto_emb.items():
                if proto is not None:
                    sem = float(cosine_similarity([sent_embs[idx]], [proto])[0][0])
                else:
                    sem = 0.0
                sem_scores[t] = sem
        else:
            for t in TOPIC_NAMES:
                sem_scores[t] = 0.0
        # final score combine (weights chosen to prefer keywords + semantics)
        combined = {}
        for t in TOPIC_NAMES:
            combined[t] = 0.7 * sem_scores.get(t, 0.0) + 0.3 * (kw_score if t in TOPIC_KEYWORDS and any(kw in s_low for kw in TOPIC_KEYWORDS[t]) else 0)
        # pick best topic
        best_topic = max(combined.items(), key=lambda x: x[1])[0]
        results[best_topic].append((s, combined[best_topic]))
    # sort sentences per topic by score desc
    for t in results:
        results[t].sort(key=lambda x: -x[1])
    return results

# ---------------- SELECT TOP SENTENCES PER TOPIC ----------------
def select_top_sentences_per_topic(classified_sentences, per_topic=6):
    slides_sentences = {}
    for t in TOPIC_NAMES:
        entries = classified_sentences.get(t, [])
        # take top N unique sentences and dedupe by similarity
        sents = [s for s, _ in entries]
        # dedupe using local method
        sents = dedupe_sentences(sents, threshold=0.84, max_keep=per_topic*2)
        slides_sentences[t] = sents[:per_topic]
    return slides_sentences

# ---------------- GEMINI CALL (polish bullets) ----------------
def _call_gemini(prompt, model_candidates=None, max_output_tokens=1024):
    if model_candidates is None:
        model_candidates = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-pro"]
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(f"google.generativeai not installed/importable: {e}")
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
        except Exception:
            pass
    last_exc = None
    for model in model_candidates:
        try:
            if hasattr(genai, "chat") and hasattr(genai.chat, "completions"):
                try:
                    cc = genai.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_output_tokens=max_output_tokens,
                        temperature=0.1
                    )
                    if hasattr(cc, "choices") and len(cc.choices):
                        message = getattr(cc.choices[0], "message", None)
                        if message:
                            content = getattr(message, "content", None)
                            if isinstance(content, str):
                                return content
                            if isinstance(content, dict):
                                for v in content.values():
                                    if isinstance(v, str):
                                        return v
                            return str(cc)
                except Exception as e:
                    last_exc = e
            if hasattr(genai, "generate"):
                try:
                    resp = genai.generate(model=model, prompt=prompt, max_output_tokens=max_output_tokens, temperature=0.1)
                    if isinstance(resp, str):
                        return resp
                    if hasattr(resp, "text") and resp.text:
                        return resp.text
                    if hasattr(resp, "candidates") and len(resp.candidates):
                        cand = resp.candidates[0]
                        return getattr(cand, "content", None) or getattr(cand, "text", None) or str(cand)
                    return str(resp)
                except Exception as e:
                    last_exc = e
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All Gemini variants failed. Last error: {last_exc}")

def polish_bullets_with_gemini(topic, sentences, max_bullets=6):
    """
    Given a topic name and supporting sentences, ask Gemini to produce up to max_bullets polished bullets.
    Returns list of bullets (complete sentences, concise).
    """
    # create a short prompt that includes supporting facts and asks for bullets only
    facts = "\n".join(f"- {s}" for s in sentences[:12])
    prompt = f"""
You are an expert presentation writer. Using ONLY the factual lines below (do NOT add new facts), produce up to {max_bullets} concise bullet points (each 8-18 words),
suitable for a presentation slide titled: "{topic}".

Facts (do not invent):
{facts}

Return strict JSON: {{ "title": "<short title (<=7 words)>", "bullets": ["b1","b2"...], "notes":"1-2 sentences" }} and nothing else.
"""
    try:
        resp = _call_gemini(prompt, max_output_tokens=400)
        # extract JSON
        m = re.search(r'(\{.*\})', resp, flags=re.S)
        blob = m.group(1) if m else resp
        obj = json.loads(blob)
        title = obj.get("title") or topic
        bullets = obj.get("bullets") or []
        notes = obj.get("notes") or ""
        # ensure bullets are full sentences and trimmed
        bullets = [b.strip() for b in bullets if len(b.strip().split()) >= 3]
        return {"title": title, "bullets": bullets[:max_bullets], "notes": notes}
    except Exception:
        # on failure, fallback to extractive bullets (use whole sentences)
        short_bullets = [s.strip() for s in sentences[:max_bullets]]
        # ensure each ends with punctuation
        short_bullets = [b if re.search(r'[.!?]$', b) else b + '.' for b in short_bullets]
        return {"title": topic, "bullets": short_bullets, "notes": ""}

# ---------------- HIGH LEVEL: build 6-topic slides from sources ----------------
def build_six_topic_slides_from_sources(sources, bullets_per_slide=5, use_gemini=True):
    """
    sources: list of {"title","url","_text","_excerpt"}
    Returns: slides (list in TOPIC_NAMES order), sources_meta
    """
    # combine texts and get sentences
    all_sents = []
    for s in sources:
        t = (s.get("_text") or "")
        sents = split_sentences(t)
        all_sents.extend(sents)
    # dedupe global
    all_sents = dedupe_sentences(all_sents, threshold=0.82, max_keep=600)
    # classify
    classified = classify_sentences_into_topics(all_sents)
    # select top sentences per topic
    selected = select_top_sentences_per_topic(classified, per_topic=bullets_per_slide * 2)
    # for each topic, pick best bullets and polish
    slides = []
    for t in TOPIC_NAMES:
        sents = selected.get(t, [])[:bullets_per_slide * 2]
        if not sents:
            slides.append({"title": t, "bullets": ["(no content found)"], "notes": ""})
            continue
        # polish if possible
        if use_gemini and GOOGLE_API_KEY:
            slide = polish_bullets_with_gemini(t, sents, max_bullets=bullets_per_slide)
        else:
            # extractive baseline: pick top N sentences and shorten if too long
            bullets = []
            for bs in sents[:bullets_per_slide]:
                b = bs.strip()
                b = re.sub(r'\s+', ' ', b)
                # ensure full sentence; if too long, try to split into clauses at semicolon or comma and pick first clause
                if len(b) > 140:
                    parts = re.split(r'[;,:] ', b)
                    b = parts[0].strip()
                    if not re.search(r'[.!?]$', b):
                        b = b + '.'
                if not re.search(r'[.!?]$', b):
                    b = b + '.'
                bullets.append(b)
            slide = {"title": t, "bullets": bullets, "notes": ""}
        slides.append(slide)
    # sources meta
    sources_meta = [{"id": f"S{i+1}", "url": s.get("url"), "title": s.get("title")} for i, s in enumerate(sources)]
    return slides, sources_meta