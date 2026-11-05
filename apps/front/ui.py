"""Shared visual language for the market and assistant workspaces."""
import streamlit as st


def apply_theme():
    st.markdown("""
    <style>
    .stApp {background:#0b1019;color:#e5edf7;}
    header[data-testid="stHeader"] {background:#0b1019;}
    .block-container {max-width:1600px;padding-top:5rem;padding-bottom:2rem;}
    h1,h2,h3 {letter-spacing:-.035em;}
    [data-testid="stMetric"] {background:#101722;border:1px solid #263043;border-radius:12px;padding:17px 20px;}
    [data-testid="stMetricLabel"] {color:#93a4bc;font-size:12px;text-transform:uppercase;letter-spacing:.07em;}
    [data-testid="stMetricValue"] {font-size:1.75rem;font-weight:600;}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#263043!important;border-radius:12px;}
    .stButton>button {border-radius:8px;border:1px solid #2b374a;background:#131d2b;color:#dbe5f1;}
    .stButton>button:hover {border-color:#32d6b3;color:#32d6b3;}
    .stButton>button[kind="primary"] {background:#32d6b3;color:#071713;border:none;font-weight:700;}
    [data-baseweb="select"]>div {background:#131d2b;color:#e5edf7;border-color:#2b374a;}
    [data-testid="stCaptionContainer"] {color:#93a4bc;}
    .brand {font-size:25px;font-weight:800;letter-spacing:-1px;margin:0 0 22px;}
    .brand span {color:#32d6b3;}
    .eyebrow {color:#32d6b3;font-size:11px;letter-spacing:.17em;font-weight:700;text-transform:uppercase;margin:10px 0;}
    .hero {background:radial-gradient(ellipse at top right,#163d3c 0%,#111b2a 45%,#101722 80%);border:1px solid #29384c;border-radius:18px;padding:48px;margin:12px 0 24px;}
    .hero h1 {font-size:52px;line-height:1.08;max-width:700px;margin:15px 0;}
    .hero p {font-size:17px;color:#a7b8cc;max-width:610px;line-height:1.7;}
    .badge {display:inline-block;color:#7ee8cf;background:#12342f;border:1px solid #235247;border-radius:6px;padding:4px 9px;font-size:11px;letter-spacing:.06em;}
    .section-label {font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:#8d9fb8;margin:14px 0;}
    .forecast-price {font-size:38px;font-weight:650;letter-spacing:-1px;margin:4px 0;color:#eaf1fa;}
    .forecast-up {color:#32d6b3;} .forecast-down {color:#f36c83;}
    footer {visibility:hidden;}
    @media(max-width:700px) {.hero {padding:24px;} .hero h1 {font-size:34px;} .block-container {padding-left:1rem;padding-right:1rem;}}
    
    [data-testid="stChatMessage"] {background:#101c2b;border:1px solid #24364a;border-radius:18px;padding:24px;margin-bottom:18px;}
    [data-testid="stChatInput"] {border:1px solid #38516b;border-radius:16px;box-shadow:0 8px 32px #0003;}
    [data-testid="stSidebar"] {background:#0e1724;border-right:1px solid #24364a;}
    [data-testid="stExpander"] {background:#101722;border-radius:12px;}
    [data-testid="stPlotlyChart"] {border-radius:12px;overflow:hidden;}
    .stApp {background:radial-gradient(ellipse at 90% 0%,#122735 0%,#0b1019 48%);}
    [data-testid="stPageLink"] {border:1px solid #2b374a;border-radius:8px;padding:3px 12px;}
    .research-heading {font-size:36px;font-weight:700;letter-spacing:-1.2px;margin:0;}
    .research-subtitle {color:#96abc2;font-size:16px;line-height:1.7;margin:8px 0 26px;}
    </style>
    """, unsafe_allow_html=True)


def apply_chat_theme():
    """A restrained conversation layout, scoped to the chat page."""
    st.markdown("""
    <style>
    .stApp, header[data-testid="stHeader"], [data-testid="stBottom"] {
        background:#0b1019;color:#e5edf7;
    }
    .block-container {max-width:850px;padding-top:5rem;padding-bottom:3rem;}
    [data-testid="stBottomBlockContainer"] {max-width:850px;padding-bottom:1rem;}
    [data-testid="stSidebar"] {background:#0e1724;border-right:0;}
    [data-testid="stSidebarNav"] {display:none;}
    [data-testid="stChatMessage"] {
        background:transparent;border:0;border-radius:0;padding:16px 0;
        margin-bottom:16px;line-height:1.75;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background:#131d2b;border-radius:24px;padding:12px 20px;
        margin-left:auto;max-width:85%;width:fit-content;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {display:none;}
    [data-testid="stChatMessageContent"] {min-width:0;}
    [data-testid="stChatInput"] {
        background:#131d2b;border:1px solid #2b374a;border-radius:26px;
        box-shadow:none;padding:8px 12px;
    }
    [data-testid="stChatInput"]:focus-within {border-color:#32d6b3;}
    [data-testid="stChatInput"] textarea {color:#e5edf7;}
    .stButton>button, [data-testid="stPageLink"] {
        background:transparent;border:1px solid #2b374a;border-radius:20px;color:#e5edf7;
    }
    .stButton>button:hover {background:#131d2b;border-color:#32d6b3;color:#32d6b3;}
    [data-testid="stSidebar"] .stButton>button,
    [data-testid="stSidebar"] [data-testid="stPageLink"] {
        border:0;border-radius:8px;text-align:left;justify-content:flex-start;
    }
    [data-testid="stSidebar"] .stButton>button[kind="primary"] {background:#12342f;color:#7ee8cf;}
    [data-testid="stExpander"] {background:transparent;border-color:#2b374a;border-radius:12px;}
    [data-testid="stCaptionContainer"] {color:#93a4bc;}
    .chat-title {font-size:20px;font-weight:600;color:#e5edf7;margin:0 0 24px;}
    .chat-title span {color:#32d6b3;}
    .chat-welcome {text-align:center;margin:18vh 0 32px;font-size:32px;font-weight:600;letter-spacing:-.7px;}
    button:focus-visible, a:focus-visible, textarea:focus-visible {outline:2px solid #32d6b3!important;outline-offset:3px;}
    /* Keep the composer focus indicator on the outer container only. */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea:focus-visible,
    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        outline:none!important;border:0!important;box-shadow:none!important;
    }
    /* Allow the sidebar column to shrink and ellipsize names, not wrap them. */
    [data-testid="stSidebar"] [data-testid="stColumn"] {min-width:0;}
    [data-testid="stSidebar"] .stButton>button {overflow:hidden;min-width:0;}
    [data-testid="stSidebar"] .stButton>button [data-testid="stMarkdownContainer"] {
        min-width:0;overflow:hidden;
    }
    [data-testid="stSidebar"] .stButton>button p {
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;
    }
    @media(max-width:700px) {
        .block-container {padding-left:1rem;padding-right:1rem;}
        .chat-welcome {margin-top:12vh;font-size:26px;}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {max-width:95%;}
    }
    </style>
    """, unsafe_allow_html=True)
