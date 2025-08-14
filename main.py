
# app.py 
import streamlit as st
from agents import search_duckduckgo, extract_article_text, build_six_topic_slides_from_sources
from ppt_generator import build_ppt_from_slides
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---- Page Config ----
st.set_page_config(
    page_title="Smart Research & PPT Builder",
    page_icon="🟣",
    layout="wide"
)

# ---- Custom CSS Styling ----
st.markdown("""
<style>
/* Background & font */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f9f5ff, #e8d6fa);
    font-family: 'Poppins', sans-serif;
    color: #3b2c4a;
    padding-top: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(199, 157, 242, 0.25);
    padding-top: 20px;
}
.sidebar-title {
    font-size: 1.4em;
    font-weight: bold;
    background: linear-gradient(90deg, #c79df2, #a675d6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 25px;
    text-align: center;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #e6ccff, #c79df2);
    color: #3b2c4a;
    font-weight: 600;
    border: none;
    padding: 10px 18px;
    border-radius: 12px;
    transition: all 0.3s ease;
    font-size: 1.05em;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #c79df2, #a675d6);
    color: white;
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 4px 14px rgba(199, 157, 242, 0.4);
}

/* Heading */
.top-heading {
    text-align: center;
    font-size: 2.8em;
    font-weight: bold;
    letter-spacing: 0.5px;
    background: linear-gradient(90deg, #a675d6, #c79df2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 35px;
}

/* Glass card with gradient border */
.glass-card {
    background: rgba(255, 255, 255, 0.72);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 4px 30px rgba(167, 117, 214, 0.15);
    margin-bottom: 20px;
    border: 1px solid rgba(199, 157, 242, 0.3);
}

/* Input box */
div[data-testid="stTextInput"] > div > input {
    background-color: white !important;
    color: #3b2c4a !important;
    border-radius: 10px;
    border: 1px solid #c79df2 !important;
    padding: 12px;
    font-size: 1.05em;
}
div[data-testid="stTextInput"] > div > input::placeholder {
    color: #9a85b8;
    opacity: 0.8;
}
            
/* ---- Alert Styling for Better Visibility ---- */
.stAlert {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* Make all alert text visible & dark */
.stAlert p, .stAlert div, .stAlert span {
    color: #3b2c4a !important;  /* Dark purple text */
    font-weight: 600 !important;
}

/* Success */
.stAlert[data-baseweb="notification"][class*="success"] {
    background-color: rgba(199, 157, 242, 0.15) !important;
    border: 1px solid rgba(167, 117, 214, 0.5) !important;
}

/* Info */
.stAlert[data-baseweb="notification"][class*="info"] {
    background-color: rgba(167, 117, 214, 0.15) !important;
    border: 1px solid rgba(167, 117, 214, 0.5) !important;
}

/* Warning */
.stAlert[data-baseweb="notification"][class*="warning"] {
    background-color: rgba(255, 204, 102, 0.2) !important;
    border: 1px solid rgba(255, 204, 102, 0.5) !important;
}

/* Error */
.stAlert[data-baseweb="notification"][class*="error"] {
    background-color: rgba(255, 102, 102, 0.2) !important;
    border: 1px solid rgba(255, 102, 102, 0.5) !important;
}


/* Hide Streamlit default menu & footer */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- Top Heading ----
st.markdown('<div class="top-heading">🟣 Smart Research & PPT Builder</div>', unsafe_allow_html=True)

# ---- Sidebar Tools ----
st.sidebar.markdown('<div class="sidebar-title">Tools</div>', unsafe_allow_html=True)
research_btn = st.sidebar.button("📄 Research Agent (Summarize)", use_container_width=True)
ppt_btn = st.sidebar.button("📑 Generate PPT", use_container_width=True)

# ---- Main Query Input ----
query = st.text_input("", placeholder="🔍 Type your research topic here...")

if query and not (research_btn or ppt_btn):
    st.info("Select a tool from the sidebar to proceed.")

# ---- Research Agent ----
if research_btn:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("🔍 Searching and summarizing..."):
            hits = search_duckduckgo(query, max_results=5)
        if not hits:
            st.error("No results found.")
        else:
            st.session_state["hits"] = hits
            
            st.markdown("### 📚 Candidate Sources")
            for h in hits:
                st.markdown(f"**{h.get('title')}**  \n{h.get('url')}")
            st.markdown('</div>', unsafe_allow_html=True)

# ---- PPT Generator ----
if ppt_btn:
    if not query.strip():
        st.warning("Please enter a query.")
    elif "hits" not in st.session_state:
        st.warning("Run Research Agent first to gather sources.")
    else:
        with st.spinner("📊 Scraping sources & generating PPT..."):
            scraped = []
            for h in st.session_state["hits"]:
                info = extract_article_text(h["url"])
                excerpt = info.get("text", "")[:500] + "..."
                scraped.append({
                    "title": info.get("title"),
                    "url": h["url"],
                    "_text": info.get("text"),
                    "_excerpt": excerpt
                })
            slides, sources_meta = build_six_topic_slides_from_sources(
                scraped,
                bullets_per_slide=5,
                use_gemini=True
            )
            ppt_path = build_ppt_from_slides(
                slides,
                presentation_title=query,
                output_path="presentation_result.pptx",
                subtitle=query,
                sources_meta=sources_meta
            )
            st.session_state["ppt_path"] = ppt_path

       
            st.success("✅ PPT Generated Successfully!")
            with open(ppt_path, "rb") as f:
                st.download_button(
                    "📥 Download PPTX",
                    f,
                    file_name="presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
            st.markdown('</div>', unsafe_allow_html=True)
