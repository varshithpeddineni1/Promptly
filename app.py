import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.rag_engine import (
    initialize_rag,
    search_similar_prompts,
    simple_search
)
from core.prompt_generator import (
    generate_prompt,
    generate_for_all_tools
)

st.set_page_config(
    page_title="Promptly",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(
            135deg, #667eea, #764ba2
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stat-card {
        background: linear-gradient(
            135deg, #667eea, #764ba2
        );
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .tip-box {
        background: #f0fff4;
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #48bb78;
        margin-bottom: 8px;
        word-wrap: break-word;
    }
    .mistake-box {
        background: #fff5f5;
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #fc8181;
        margin-bottom: 8px;
        word-wrap: break-word;
    }
    .similar-card {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #667eea;
        margin-bottom: 8px;
        word-wrap: break-word;
    }
    textarea {
        font-family: monospace !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    .stTextArea textarea {
        background-color: #f8f9fa !important;
        color: #1a1a2e !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── INITIALIZE RAG ─────────────────────
@st.cache_resource
def load_rag_system():
    return initialize_rag()

# ── TOOL DEFINITIONS ───────────────────
TOOLS = {
    "🎨 Midjourney": "midjourney",
    "🖼️ DALL-E 3": "dalle",
    "🎭 Stable Diffusion": "stable_diffusion",
    "💬 ChatGPT": "chatgpt",
    "🤖 Claude": "claude",
    "🌟 Gemini": "gemini",
    "🎬 Sora": "sora",
    "🎥 Runway": "runway",
    "💻 Copilot": "copilot",
    "⚡ Cursor": "cursor"
}

# ── HEADER ─────────────────────────────
st.markdown(
    '<div class="main-title">⚡ Promptly</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Perfect Prompts for Every AI — Instantly</div>',
    unsafe_allow_html=True
)

st.divider()

# ── LOAD RAG ───────────────────────────
with st.spinner("🧠 Loading AI knowledge base..."):
    collection, all_prompts = load_rag_system()

# ── STATS BAR ──────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <h2>10+</h2>
        <p>AI Tools Supported</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <h2>{len(all_prompts)}+</h2>
        <p>Expert Prompts</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <h2>RAG</h2>
        <p>Powered Technology</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <h2>100%</h2>
        <p>Free to Use</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── SIDEBAR ────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    generation_mode = st.radio(
        "Generation Mode:",
        ["Single Tool", "Multiple Tools"],
        horizontal=True
    )

    style = st.select_slider(
        "Style:",
        options=["minimal", "balanced", "detailed"],
        value="balanced"
    )

    detail_level = st.select_slider(
        "Detail Level:",
        options=["low", "medium", "high"],
        value="high"
    )

    tone = st.selectbox(
        "Tone:",
        ["professional", "creative",
         "technical", "casual", "artistic"]
    )

    st.divider()

    st.markdown("### 📚 Quick Examples")

    examples = [
        "A logo for a tech startup",
        "Explain machine learning simply",
        "A cinematic sunset landscape",
        "Write a viral LinkedIn post",
        "Debug this Python code",
        "A futuristic city at night",
        "Create a marketing email",
        "An anime character portrait"
    ]

    for example in examples:
        if st.button(
            f"💡 {example[:30]}",
            use_container_width=True
        ):
            st.session_state.user_input = example

    st.divider()

    if 'saved_prompts' not in st.session_state:
        st.session_state.saved_prompts = []

    st.markdown(
        f"### ⭐ Saved Prompts "
        f"({len(st.session_state.saved_prompts)})"
    )

    for i, saved in enumerate(
        st.session_state.saved_prompts[-5:]
    ):
        with st.expander(f"📝 {saved['tool']}"):
            st.text_area(
                "",
                value=saved['prompt'][:200],
                height=100,
                label_visibility="collapsed",
                key=f"saved_{i}"
            )

# ── MAIN INPUT ─────────────────────────
st.markdown("### 💭 What Do You Want to Create?")

user_input = st.text_area(
    "",
    value=st.session_state.get('user_input', ''),
    placeholder="Describe what you want to create in plain English...\n\nExample: A professional headshot photo of a young entrepreneur in a modern office setting",
    height=120,
    label_visibility="collapsed"
)

# ── TOOL SELECTION ─────────────────────
st.markdown("### 🛠️ Select AI Tool(s)")

if generation_mode == "Single Tool":
    selected_tool_display = st.selectbox(
        "Choose your AI tool:",
        list(TOOLS.keys()),
        label_visibility="collapsed"
    )
    selected_tools = [TOOLS[selected_tool_display]]
else:
    cols = st.columns(5)
    selected_tools = []

    for i, (tool_display, tool_value) in enumerate(
        TOOLS.items()
    ):
        with cols[i % 5]:
            if st.checkbox(tool_display, value=i < 3):
                selected_tools.append(tool_value)

# ── GENERATE BUTTON ────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate = st.button(
        "⚡ Generate Perfect Prompts",
        use_container_width=True,
        type="primary"
    )

# ── GENERATION LOGIC ───────────────────
if generate and user_input and selected_tools:

    with st.spinner("🔍 Searching knowledge base..."):
        if collection:
            similar = search_similar_prompts(
                user_input,
                collection,
                n_results=5
            )
        else:
            similar = simple_search(
                user_input,
                all_prompts,
                n=5
            )

    if similar:
        with st.expander(
            f"📚 Found {len(similar)} similar expert prompts"
        ):
            for sp in similar:
                st.markdown(f"""
                <div class="similar-card">
                    <strong>{sp['tool'].upper()}</strong> —
                    {sp['description']}<br>
                    <small>Relevance: {sp['relevance']}</small>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.markdown("## ✨ Your Generated Prompts")

    for tool in selected_tools:
        tool_display = [
            k for k, v in TOOLS.items()
            if v == tool
        ]
        tool_name = tool_display[0] \
            if tool_display else tool

        with st.spinner(
            f"⚡ Generating {tool_name} prompt..."
        ):
            result = generate_prompt(
                user_input,
                tool,
                similar,
                style=style,
                detail_level=detail_level,
                tone=tone
            )

        if result["success"]:
            data = result["data"]

            st.markdown(f"### {tool_name}")

            # ── MAIN PROMPT ────────────
            st.markdown("**⚡ Your Optimized Prompt:**")

            main_prompt = data.get('main_prompt', '')

            # Display in readable text area
            st.text_area(
                "",
                value=main_prompt,
                height=180,
                label_visibility="collapsed",
                key=f"main_{tool}"
            )

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "📋 Copy",
                    key=f"copy_{tool}",
                    use_container_width=True
                ):
                    st.success("✅ Copied to clipboard!")

            with col2:
                if st.button(
                    "⭐ Save",
                    key=f"save_{tool}",
                    use_container_width=True
                ):
                    st.session_state.saved_prompts.append({
                        'tool': tool_name,
                        'prompt': main_prompt,
                        'description': user_input
                    })
                    st.success("✅ Saved!")

            with col3:
                if st.button(
                    "🔄 Regenerate",
                    key=f"regen_{tool}",
                    use_container_width=True
                ):
                    st.rerun()

            # ── TABS ───────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "📖 Explanation",
                "🔀 Variations",
                "💡 Tips",
                "⚠️ Avoid"
            ])

            with tab1:
                st.info(
                    data.get(
                        'explanation',
                        'No explanation available'
                    )
                )

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📝 Detailed Version:**")
                    st.text_area(
                        "",
                        value=data.get(
                            'variation_detailed', ''
                        ),
                        height=150,
                        label_visibility="collapsed",
                        key=f"detail_{tool}"
                    )
                with col2:
                    st.markdown("**✂️ Simple Version:**")
                    st.text_area(
                        "",
                        value=data.get(
                            'variation_simple', ''
                        ),
                        height=150,
                        label_visibility="collapsed",
                        key=f"simple_{tool}"
                    )

            with tab3:
                tips = data.get('tips', [])
                for tip in tips:
                    st.markdown(f"""
                    <div class="tip-box">
                        💡 {tip}
                    </div>
                    """, unsafe_allow_html=True)

            with tab4:
                mistakes = data.get(
                    'mistakes_to_avoid', []
                )
                for mistake in mistakes:
                    st.markdown(f"""
                    <div class="mistake-box">
                        ⚠️ {mistake}
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

        else:
            st.error(
                f"Failed to generate {tool_name}: "
                f"{result.get('error', 'Unknown error')}"
            )

elif generate and not user_input:
    st.error("Please describe what you want to create!")

elif generate and not selected_tools:
    st.error("Please select at least one AI tool!")

# ── FOOTER ─────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("⚡ **Promptly**")
with col2:
    st.markdown(
        "Built by **Varshith Peddineni**"
    )
with col3:
    st.markdown("Powered by **Groq + RAG**")
    