import json
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Startup Growth Advisor", page_icon="🚀", layout="wide")

# Works on both Windows (local) and Linux (Render), from any working directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fourth_new_et_model_n_100_pipe.pkl"

# =====================================================================
#  API KEY GATE (naya): key daale bina aage nahi badh sakte
# =====================================================================

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

/* page background */
.stApp {
    background:
        radial-gradient(1000px 500px at 85% -10%, rgba(99,102,241,0.18), transparent 60%),
        radial-gradient(800px 400px at -5% 10%, rgba(236,72,153,0.12), transparent 60%),
        #0b1020;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1180px; }

/* global text colours for dark look */
.stApp, .stApp p, .stApp label, .stApp span, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #e5e7ef; }
label, .stMarkdown small, [data-testid="stCaptionContainer"] { color: #9aa3b8 !important; }

/* hero */
.hero {
    position: relative; overflow: hidden;
    padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    box-shadow: 0 20px 50px rgba(79,70,229,0.35);
}
.hero::after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 70%); border-radius: 50%;
}
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; color: #fff !important; letter-spacing: -0.5px; }
.hero p { margin: 0.5rem 0 0; font-size: 1.02rem; color: rgba(255,255,255,0.88) !important; }
.pill {
    display: inline-block; margin: 0.9rem 0.4rem 0 0; padding: 0.28rem 0.8rem; border-radius: 999px;
    background: rgba(255,255,255,0.18); color: #fff !important; font-size: 0.78rem; font-weight: 600;
    backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.25);
}

/* section titles */
.sec-title {
    display: flex; align-items: center; gap: 0.6rem; margin: 0.4rem 0 0.9rem;
    font-size: 1.12rem; font-weight: 700; color: #fff;
}
.sec-title .num {
    width: 28px; height: 28px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6366f1, #a855f7); font-size: 0.85rem; color: #fff;
}

/* form card */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px; padding: 1.6rem 1.8rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}

/* inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important; color: #f1f3fa !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: #8b5cf6 !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.25) !important;
}
div[data-baseweb="select"] span { color: #f1f3fa !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,0.04); padding: 6px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 0.5rem 1.1rem; background: transparent; color: #9aa3b8; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 14px; font-weight: 700; padding: 0.7rem 1.2rem; border: 0;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899); color: #fff;
    box-shadow: 0 10px 28px rgba(139,92,246,0.4); transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px); box-shadow: 0 14px 34px rgba(139,92,246,0.55); color: #fff;
}

/* expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.09); border-radius: 14px;
}

/* metric cards */
.kpi-wrap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 0.6rem 0 0.4rem; }
.kpi {
    position: relative; overflow: hidden; border-radius: 18px; padding: 1.3rem 1.4rem;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--accent); }
.kpi .lbl { font-size: 0.78rem; letter-spacing: .08em; text-transform: uppercase; color: #9aa3b8; font-weight: 600; }
.kpi .val { font-size: 1.85rem; font-weight: 800; color: #fff; margin-top: 0.3rem; letter-spacing: -0.5px; }
.kpi .sub { font-size: 0.82rem; margin-top: 0.2rem; color: #9aa3b8; }
.kpi.up   { --accent: #22c55e; } .kpi.up .val { color: #4ade80; }
.kpi.down { --accent: #ef4444; } .kpi.down .val { color: #f87171; }
.kpi.cur  { --accent: #38bdf8; }
.kpi.prd  { --accent: #a855f7; }
@media (max-width: 800px) { .kpi-wrap { grid-template-columns: 1fr; } }

/* chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
}
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stChatInput"] textarea { color: #f1f3fa !important; }

hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

if not st.session_state.groq_api_key:
    st.markdown(
        """
<div class="hero">
    <h1>🚀 Startup Growth Advisor</h1>
    <p>Enter your Groq API key to unlock the app.</p>
    <span class="pill">🔒 Key is kept only in this browser session</span>
</div>
""",
        unsafe_allow_html=True,
    )
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        with st.form("api_key_form"):
            st.markdown("#### 🔑 Groq API key")
            entered_key = st.text_input("API key", type="password", placeholder="gsk_...",
                                        label_visibility="collapsed")
            unlock = st.form_submit_button("Unlock app", type="primary", use_container_width=True)
        if unlock:
            if not entered_key.strip():
                st.error("Please enter your Groq API key to continue.")
            else:
                with st.spinner("Checking your key..."):
                    try:
                        ChatGroq(model='openai/gpt-oss-120b', api_key=entered_key.strip()).invoke("hi")
                    except Exception:
                        st.error("This key did not work. Please check it and try again.")
                    else:
                        st.session_state.groq_api_key = entered_key.strip()
                        st.rerun()
    st.stop()   # key nahi hai to yahin ruk jao, neeche ka kuch nahi chalega

# =====================================================================
#  YOUR LOGIC (unchanged except model loading)
# =====================================================================

GROQ_API_KEY = st.session_state.groq_api_key
llm = ChatGroq(model='openai/gpt-oss-120b', api_key=GROQ_API_KEY)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}. "
                 "Make sure models/fourth_new_et_model_n_100_pipe.pkl is pushed to GitHub.")
        st.stop()
    return joblib.load(MODEL_PATH)


fourth_new_et_model_n_100_pipe = load_model()   # loaded once, reused on every rerun


def feature_engineering(df):

    # Convert date columns
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['founded_date'] = pd.to_datetime(df['founded_date'], errors='coerce')
    df['incubation_start_date'] = pd.to_datetime(df['incubation_start_date'], errors='coerce')
    df['incubation_end_date'] = pd.to_datetime(df['incubation_end_date'], errors='coerce')

    TODAY = pd.Timestamp(df['created_at'].max()).normalize()
    df['has_gst'] = df['gst_number'].notna().astype(int)  # not null

    # calculates months in incubation
    df['months_in_incubation'] = ((TODAY - pd.to_datetime(df['incubation_start_date'])).dt.days / 30.4).clip(lower=0)

    df['months_left'] = ((pd.to_datetime(df['incubation_end_date']) - TODAY).dt.days / 30.4).clip(lower=0)

    df = df.drop([
        'id', 'organization_id', 'logo_url', 'website',
        'founder_id', 'founder_email', 'founder_phone', 'founder_name',
        'registration_number', 'pan_number', 'gst_number', 'cin_number',
        'pitch_deck_url', 'business_plan_url',
        'status', 'is_active',
        'created_at', 'updated_at', 'deleted_at', 'created_by',
        'incubation_start_date', 'incubation_end_date', 'team_members'
    ], axis=1)

    df['age_months'] = ((TODAY - df['founded_date']).dt.days / 30.4).clip(lower=0)
    df['cofounder_count'] = df['cofounders'].apply(lambda s: len(json.loads(s)))  # count members in cofounders list
    df = df.drop(['founded_date', 'cofounders'], axis=1)

    # business_model -> 5 groups
    bm_map = {
        'B2B logistics platform': 'B2B', 'B2B SaaS': 'B2B', 'B2B API': 'B2B',
        'B2C subscription': 'B2C', 'B2C app': 'B2C', 'D2C brand': 'B2C',
        'B2B2C platform': 'Mixed', 'B2C and B2B platform': 'Mixed', 'B2C and B2B SaaS': 'Mixed',
        'B2B and B2C SaaS': 'Mixed', 'B2C and B2B app': 'Mixed', 'B2C app and B2B partnerships': 'Mixed',
        'B2B marketplace': 'Marketplace', 'Online marketplace': 'Marketplace', 'M2C marketplace': 'Marketplace',
        'Digital lending marketplace': 'Marketplace', 'Service and marketplace': 'Marketplace',
        'Hardware plus SaaS': 'Hardware_Service', 'Hardware plus subscription': 'Hardware_Service',
        'Service and financing': 'Hardware_Service', 'Space as a service': 'Hardware_Service',
        'Service plus technology': 'Hardware_Service'}

    df['buisness_group'] = df['business_model'].map(bm_map)

    # ['Idea'  'MVP' 'Pre-revenue' 'Revenue''Growth' 'Scale']
    df['stage'] = df['stage'].map({
        'Idea': 0,
        'MVP': 1,
        'Pre-revenue': 2,
        'Revenue': 3,
        'Growth': 4,
        'Scale': 5
    })

    df['lifecycle_stage'] = df['lifecycle_stage'].map({
        'application': 0,
        'incubation': 1
    })

    revenue_map = {
        # ---- Subscription ----
        'Per-user monthly subscription': 'Subscription',
        'Weekly and monthly meal plans': 'Subscription',
        'Device sale and monthly subscription': 'Subscription',
        'Subscription plus usage fees': 'Subscription',
        'Subscription plans': 'Subscription',
        'Monthly subscription': 'Subscription',
        'Monthly plan plus transaction fee': 'Subscription',
        'Platform commission and clinic subscriptions': 'Subscription',
        'Yearly licence per employee': 'Subscription',
        'Institutional licences and optional subscriptions': 'Subscription',
        'Subscription and input-seller commission': 'Subscription',
        'Per-employee monthly fee': 'Subscription',
        'Monthly patient subscription': 'Subscription',
        'Tiered subscription': 'Subscription',

        # ---- Commission ----
        'Commission per trip': 'Commission',
        'Commission on sales': 'Commission',
        'Listing fees and service commission': 'Commission',
        'Commission and shipping fees': 'Commission',
        'Lab commission and premium plans': 'Commission',
        'Margin on produce sold': 'Commission',
        'Loan processing commission': 'Commission',

        # ---- Sales ----
        'Device sale and yearly service plan': 'Sales',
        'Product sales': 'Sales',
        'Sale of recycled material and pickup fees': 'Sales',

        # ---- Fees ----
        'Fee per delivery': 'Fees',
        'Course fees and placement fees': 'Fees',
        'Fee per verification': 'Fees',
        'Project fees and maintenance contracts': 'Fees',
        'Desk and room rentals': 'Fees',
        'Fee per shipment and monthly contracts': 'Fees',
        'Fee per transaction and premium plan': 'Fees',
    }

    df['revenue_model'] = df['revenue_model'].map(revenue_map)

    return df


model_cols = [
    'team_size', 'annual_revenue', 'funding_raised',
    'customer_count', 'mrr', 'stage', 'lifecycle_stage',
    'has_gst', 'months_in_incubation', 'months_left',
    'age_months', 'cofounder_count', 'sector',
    'buisness_group', 'revenue_model', 'registration_type'
]


# picked the new tunded et model with n_estimators=100 because the test rmse test r2 was better than the previous models..
def model_prediction(input_df):

    model_df = feature_engineering(input_df.copy())
    model_df = model_df[model_cols]

    revenue_now = input_df['annual_revenue'][0]
    pred = np.expm1(fourth_new_et_model_n_100_pipe.predict(model_df)[0])
    growth = (pred - revenue_now) / revenue_now * 100
    print(f"  Annual revenue : ₹{revenue_now}")
    print(f"  Predicted revenue after 12 months : ₹{pred:,.0f}")
    print(f"  Predicted growth                  : {growth:.1f}%")

    return pred, growth


def prepare_startup_details(input_df):

    excluded_columns = [
        'id',
        'organization_id',
        'founder_id',
        'founder_email',
        'founder_phone',
        'founder_name',
        'registration_number',
        'pan_number',
        'gst_number',
        'cin_number',
        'logo_url',
        'website',
        'pitch_deck_url',
        'business_plan_url',
        'created_by',
        'cofounders',
        'team_members',
        'updated_at',
        'deleted_at'
    ]

    startup_df = input_df.drop(
        columns=excluded_columns,
        errors='ignore'
    )

    return startup_df.to_dict(orient='records')[0]  # will get the dictionary of the input_Df


# chat memory store (kept in session_state so it survives Streamlit reruns)
if "store" not in st.session_state:
    st.session_state.store = {}
store = st.session_state.store


def get_session_history(session_id):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI startup growth advisor.

Your job is to analyze startup information, explain the ML model's revenue
prediction, and answer the founder's follow-up questions.

Rules:
- The revenue prediction comes from a trained ML model.
- Never change or invent the predicted revenue.
- Never invent financial information.
- Clearly distinguish model predictions from your own suggestions.
- For follow-up questions, answer only what is asked, keep it practical.

STARTUP DETAILS:
{startup_details}

ML MODEL PREDICTIONS:
- Predicted annual revenue after 12 months: ₹{predicted_revenue}
- Predicted growth: {predicted_growth}%"""),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

chain = prompt | llm

chatbot_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# =====================================================================
#  STREAMLIT UI  (only the look & layout changed — same inputs/outputs)
# =====================================================================

BUSINESS_MODELS = [
    'B2B logistics platform', 'B2B SaaS', 'B2B API',
    'B2C subscription', 'B2C app', 'D2C brand',
    'B2B2C platform', 'B2C and B2B platform', 'B2C and B2B SaaS',
    'B2B and B2C SaaS', 'B2C and B2B app', 'B2C app and B2B partnerships',
    'B2B marketplace', 'Online marketplace', 'M2C marketplace',
    'Digital lending marketplace', 'Service and marketplace',
    'Hardware plus SaaS', 'Hardware plus subscription',
    'Service and financing', 'Space as a service', 'Service plus technology',
]

REVENUE_MODELS = [
    'Per-user monthly subscription', 'Weekly and monthly meal plans',
    'Device sale and monthly subscription', 'Subscription plus usage fees',
    'Subscription plans', 'Monthly subscription', 'Monthly plan plus transaction fee',
    'Platform commission and clinic subscriptions', 'Yearly licence per employee',
    'Institutional licences and optional subscriptions',
    'Subscription and input-seller commission', 'Per-employee monthly fee',
    'Monthly patient subscription', 'Tiered subscription',
    'Commission per trip', 'Commission on sales', 'Listing fees and service commission',
    'Commission and shipping fees', 'Lab commission and premium plans',
    'Margin on produce sold', 'Loan processing commission',
    'Device sale and yearly service plan', 'Product sales',
    'Sale of recycled material and pickup fees',
    'Fee per delivery', 'Course fees and placement fees', 'Fee per verification',
    'Project fees and maintenance contracts', 'Desk and room rentals',
    'Fee per shipment and monthly contracts', 'Fee per transaction and premium plan',
]

STAGES = ['Idea', 'MVP', 'Pre-revenue', 'Revenue', 'Growth', 'Scale']
LIFECYCLES = ['application', 'incubation']
REG_TYPES = ['Private Limited', 'LLP', 'Partnership', 'Proprietorship', 'One Person Company', 'Public Limited']

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "ready" not in st.session_state:
    st.session_state.ready = False


def ask(question):
    response = chatbot_with_memory.invoke(
        {
            "startup_details": st.session_state.startup_details,
            "predicted_revenue": f"{st.session_state.pred:,.0f}",
            "predicted_growth": f"{st.session_state.growth:.1f}",
            "question": question,
        },
        config={"configurable": {"session_id": st.session_state.session_id}},
    )
    return response.content


# ---------------------------- STYLING ----------------------------


def section(num, title):
    st.markdown(
        f'<div class="sec-title"><span class="num">{num}</span>{title}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------- HERO ----------------------------
st.markdown(
    """
<div class="hero">
    <h1>🚀 Startup Growth Advisor</h1>
    <p>Fill in your startup details, get an ML-powered 12-month revenue forecast, then chat with your AI growth advisor.</p>
    <span class="pill">📈 ML Revenue Prediction</span>
    <span class="pill">🤖 AI Advisor Chat</span>
    <span class="pill">💡 Practical Growth Tips</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------- INPUT FORM ----------------------------
with st.form("startup_form"):

    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Startup", "👥 Team", "💰 Financials", "📅 Dates & Registration"])

    with tab1:
        section(1, "Startup details")
        c1, c2, c3 = st.columns(3)
        startup_name = c1.text_input("Startup name", "GreenBasket")
        tagline_pitch = c2.text_input("Tagline", "Fresh groceries delivered to your doorstep")
        sector = c3.text_input("Sector", "AgriTech")

        description = st.text_area("Description", "An online grocery delivery platform connecting local farmers with customers.")

        c1, c2, c3 = st.columns(3)
        stage = c1.selectbox("Stage", STAGES, index=STAGES.index('Growth'))
        lifecycle_stage = c2.selectbox("Lifecycle stage", LIFECYCLES, index=1)
        registration_type = c3.selectbox("Registration type", REG_TYPES)

        c1, c2, c3 = st.columns(3)
        business_model = c1.selectbox("Business model", BUSINESS_MODELS, index=BUSINESS_MODELS.index('M2C marketplace'))
        revenue_model = c2.selectbox("Revenue model", REVENUE_MODELS, index=REVENUE_MODELS.index('Commission per trip'))
        target_market = c3.text_input("Target market", "Urban households and local farmers")

        c1, c2 = st.columns(2)
        problem_statement = c1.text_area("Problem statement", "Customers struggle to find fresh and affordable farm produce.")
        solution_description = c2.text_area("Solution description", "A digital marketplace connecting farmers directly with customers.")

    with tab2:
        section(2, "Team")
        c1, c2, c3 = st.columns(3)
        founder_name = c1.text_input("Founder name", "Amit Sharma")
        founder_email = c2.text_input("Founder email", "founder@greenbasket.com")
        founder_phone = c3.text_input("Founder phone", "919876543210")

        c1, c2, c3 = st.columns(3)
        team_size = c1.number_input("Team size", min_value=0, value=8, step=1)
        cofounders_txt = c2.text_area("Co-founders (one name per line)", "Rahul Sharma")
        team_members_txt = c3.text_area("Team members (one name per line)", "Priya Shah")

    with tab3:
        section(3, "Financials")
        c1, c2, c3 = st.columns(3)
        monthly_revenue = c1.number_input("Monthly revenue (₹)", min_value=0, value=250000, step=1000)
        annual_revenue = c2.number_input("Annual revenue (₹)", min_value=1, value=3000000, step=10000)
        mrr = c3.number_input("MRR (₹)", min_value=0, value=250000, step=1000)

        c1, c2 = st.columns(2)
        funding_raised = c1.number_input("Funding raised (₹)", min_value=0, value=1500000, step=10000)
        customer_count = c2.number_input("Customer count", min_value=0, value=1200, step=1)

    with tab4:
        section(4, "Dates")
        c1, c2, c3 = st.columns(3)
        founded_date = c1.date_input("Founded date", datetime(2024, 6, 15))
        incubation_start_date = c2.date_input("Incubation start date", datetime(2025, 1, 1))
        incubation_end_date = c3.date_input("Incubation end date", datetime(2026, 12, 31))

        with st.expander("5. Registration & other details (optional)"):
            c1, c2 = st.columns(2)
            registration_number = c1.text_input("Registration number", "U12345MH2024PTC123456")
            cin_number = c2.text_input("CIN number", "U12345MH2024PTC123456")
            c1, c2 = st.columns(2)
            pan_number = c1.text_input("PAN number", "ABCDE1234F")
            gst_number = c2.text_input("GST number (leave blank if none)", "27ABCDE1234F1Z5")
            c1, c2 = st.columns(2)
            website = c1.text_input("Website", "https://greenbasket.example.com")
            logo_url = c2.text_input("Logo URL", "https://example.com/greenbasket-logo.png")
            c1, c2 = st.columns(2)
            pitch_deck_url = c1.text_input("Pitch deck URL", "https://example.com/pitch-deck.pdf")
            business_plan_url = c2.text_input("Business plan URL", "https://example.com/business-plan.pdf")

    submitted = st.form_submit_button("🔮 Predict & Start Chat", type="primary", use_container_width=True)

if submitted:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cofounders = json.dumps([{"name": n.strip(), "role": "Co-founder"} for n in cofounders_txt.splitlines() if n.strip()])
    team_members = json.dumps([{"name": n.strip(), "role": "Team Member"} for n in team_members_txt.splitlines() if n.strip()])

    input_df = pd.DataFrame({
        'id': [1],
        'organization_id': ['1a2a73ed-562b-4f79-8374-59eef50bea63'],
        'startup_name': [startup_name],
        'tagline_pitch': [tagline_pitch],
        'logo_url': [logo_url],
        'website': [website],
        'description': [description],
        'sector': [sector],
        'stage': [stage],
        'founded_date': [str(founded_date)],
        'founder_id': ['5be6128e-18c2-4797-a142-ea7d17be3111'],
        'founder_email': [founder_email],
        'founder_phone': [founder_phone],
        'cofounders': [cofounders],
        'team_members': [team_members],
        'team_size': [int(team_size)],
        'problem_statement': [problem_statement],
        'solution_description': [solution_description],
        'target_market': [target_market],
        'business_model': [business_model],
        'revenue_model': [revenue_model],
        'registration_type': [registration_type],
        'registration_number': [registration_number],
        'pan_number': [pan_number],
        'gst_number': [gst_number.strip() if gst_number.strip() else None],
        'cin_number': [cin_number],
        'monthly_revenue': [int(monthly_revenue)],
        'annual_revenue': [int(annual_revenue)],
        'funding_raised': [int(funding_raised)],
        'customer_count': [int(customer_count)],
        'mrr': [int(mrr)],
        'pitch_deck_url': [pitch_deck_url],
        'business_plan_url': [business_plan_url],
        'status': ['active'],
        'lifecycle_stage': [lifecycle_stage],
        'incubation_start_date': [str(incubation_start_date)],
        'incubation_end_date': [str(incubation_end_date)],
        'is_active': [True],
        'created_at': [now],
        'updated_at': [now],
        'deleted_at': [None],
        'created_by': ['6310c3f5-f5c2-55f1-ab4b-ab0798a6a89b'],
        'founder_name': [founder_name]
    })

    with st.spinner("Running ML model..."):
        pred, growth = model_prediction(input_df)
        startup_details = prepare_startup_details(input_df)

    st.session_state.pred = pred
    st.session_state.growth = growth
    st.session_state.startup_details = startup_details
    st.session_state.current_revenue = int(annual_revenue)
    st.session_state.startup_name = startup_name
    st.session_state.session_id = str(uuid.uuid4())  # fresh chat memory for new startup
    st.session_state.messages = []
    st.session_state.ready = True

# ---------------------------- RESULT + CHAT ----------------------------
if st.session_state.ready:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-title" style="font-size:1.4rem;">📊 ML Prediction for {st.session_state.startup_name}</div>',
        unsafe_allow_html=True,
    )

    growth_cls = "up" if st.session_state.growth >= 0 else "down"
    arrow = "▲" if st.session_state.growth >= 0 else "▼"

    st.markdown(
        f"""
<div class="kpi-wrap">
    <div class="kpi cur">
        <div class="lbl">Current annual revenue</div>
        <div class="val">₹{st.session_state.current_revenue:,.0f}</div>
        <div class="sub">Today</div>
    </div>
    <div class="kpi prd">
        <div class="lbl">Predicted revenue (12 months)</div>
        <div class="val">₹{st.session_state.pred:,.0f}</div>
        <div class="sub">Model forecast</div>
    </div>
    <div class="kpi {growth_cls}">
        <div class="lbl">Predicted growth</div>
        <div class="val">{arrow} {st.session_state.growth:.1f}%</div>
        <div class="sub">vs. current revenue</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title" style="font-size:1.4rem;">💬 Chat with your AI Growth Advisor</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        if st.button("📋 Get structured analysis of my startup"):
            q = "Give a structured analysis of this startup."
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                st.session_state.messages.append({"role": "assistant", "content": ask(q)})
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything, e.g. How can I increase my customer base?")
    if question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                answer = ask(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
