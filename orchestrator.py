# orchestrator.py
from agents import extract_article_text, build_topic_slides_from_sources, top_k_by_tfidf

def orchestrate_scrape_and_preview(urls):
    """
    Scrape URLs and produce preview entries with concise excerpt (1-2 sentences).
    Returns list of dicts: {title,url,_text,_excerpt}
    """
    sources = []
    for u in urls:
        info = extract_article_text(u)
        text = info.get("text", "")
        title = info.get("title") or u
        top = top_k_by_tfidf(split_sentences := text and text.split(".") or [], k=2)  # quick fallback
        excerpt = (" ".join(top)).strip()
        if not excerpt:
            excerpt = (text[:300] + "...") if text else ""
        sources.append({"title": title, "url": u, "_text": text, "_excerpt": excerpt})
    return sources

def orchestrate_generate_topic_ppt(query, selected_sources, max_topics=5, output_ppt="result.pptx", use_gemini=True):
    """
    Build slides by semantic topics from selected_sources (list of source dicts containing _text).
    Returns: slides, ppt_path, clusters
    """
    # build topic slides
    slides, clusters = build_topic_slides_from_sources(selected_sources, max_topics=max_topics, use_gemini=use_gemini)
    # create presentation via ppt_generator
    from ppt_generator import build_ppt_from_slides
    presentation_title = query
    subtitle = "Auto-generated — topic-based slides"
    ppt_path = build_ppt_from_slides(slides, presentation_title, output_path=output_ppt, subtitle=subtitle, sources_meta=[{"id": f"S{i+1}", "url": s.get("url")} for i, s in enumerate(selected_sources)])
    return slides, ppt_path, clusters