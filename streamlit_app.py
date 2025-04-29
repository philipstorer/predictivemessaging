import streamlit as st
import openai
import pandas as pd
import plotly.graph_objects as go
import json
import re
import time

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Predictive Message Testing Dashboard", layout="wide")

# Style for cards
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    border: 1px solid #d3d3d3;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
}
.section-title {
    font-size:22px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><h1>Predictive Message Testing Dashboard</h1><p>Advanced Cognitive-Linguistic Diagnostic and Optimization</p></div>', unsafe_allow_html=True)

def call_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
        max_tokens=3500
    )
    return response.choices[0].message.content

def extract_json_block(response_text, label):
    try:
        match = re.search(rf"{label}:\s*(\{{.*?\}})", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            return None
    except Exception:
        return None

def extract_improved_message(response_text):
    try:
        match = re.search(r'Improved_Message:\s*"(.*?)"', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r'Improved_Message:\s*(.*?)\s*Scores_JSON:', response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return None
    except Exception:
        return None

# Form Card
with st.form("input_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Message Input</div>', unsafe_allow_html=True)
    original_message = st.text_area("Enter Original Message:", height=150)
    persona = st.selectbox("Select Target Persona:", [
        "Women 45-65, Caregiving Role",
        "Female Millennials, Career-Focused",
        "Female Patients, Chronic Illness Management",
        "Female Healthcare Providers, Clinical Setting"
    ])
    tone = st.selectbox("Desired Tone:", ["Empathetic", "Clinical", "Inspirational", "Direct"], index=0)
    submit_button = st.form_submit_button("Analyze Message")
    st.markdown('</div>', unsafe_allow_html=True)

if submit_button and original_message:
    with st.spinner('Starting cognitive-linguistic analysis...'):
        spinner_messages = [
            "Evaluating Relational Anchoring...",
            "Assessing Emotional Reality Validation...",
            "Reviewing Narrative Integration...",
            "Measuring Collaborative Agency Framing...",
            "Checking Value-Embedded Motivation...",
            "Analyzing Cognitive Effort Reduction...",
            "Assessing Temporal Emotional Framing...",
            "Evaluating Empathic Leadership Positioning...",
            "Reviewing Affective Modality Matching..."
        ]
        for message in spinner_messages:
            st.markdown(f'<div class="card">{message}</div>', unsafe_allow_html=True)
            time.sleep(0.7)

    original_length = len(original_message)

    system_prompt_original = f"""
You are a senior communication strategist specializing in psycholinguistics.

Evaluate the following ORIGINAL MESSAGE according to the Cognitive-Linguistic Deep Analysis Model.
Persona: {persona}
Tone: {tone}

Perform:
- Output a Markdown table:
    | Domain | Score (0-10) | Diagnostic Insight | Strategic Impact |
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary
- THEN suggest an improved version.

Constraints:
- Preserve explicit claims exactly.
- Maintain intent.
- Improve clarity and tone.
- Stay within ±15% character count ({original_length} characters).

Clearly output:
Improved_Message:
Scores_JSON:
"""
    original_response = call_gpt(system_prompt_original)
    improved_message = extract_improved_message(original_response)
    original_scores = extract_json_block(original_response, "Scores_JSON")

    improved_response = ""
    improved_scores = {}

    if improved_message:
        system_prompt_improved = f"""
You are a senior communication strategist specializing in psycholinguistics.

Evaluate the following IMPROVED MESSAGE according to the same model.
Persona: {persona}
Tone: {tone}

Clearly output:
Scores_JSON:
"""
        improved_response = call_gpt(system_prompt_improved)
        improved_scores = extract_json_block(improved_response, "Scores_JSON")

    # Display Original Message
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Original Message")
    st.write(original_message)
    st.markdown('</div>', unsafe_allow_html=True)

    # Evaluation Key
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Evaluation Criteria Definitions")
    st.markdown("""
- **Relational Anchoring**: Connects to relationships.
- **Emotional Reality Validation**: Validates emotional states.
- **Narrative Integration**: Weaves into storytelling.
- **Collaborative Agency Framing**: Supports choice and partnership.
- **Value-Embedded Motivation**: Ties action to values.
- **Cognitive Effort Reduction**: Easy to understand and act.
- **Temporal Emotional Framing**: Creates urgency appropriately.
- **Empathic Leadership Positioning**: Leads empathetically.
- **Affective Modality Matching**: Matches emotional communication style.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    if original_scores:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Original Evaluation Scores")
        cols = st.columns(3)
        for idx, (domain, score) in enumerate(original_scores.items()):
            with cols[idx % 3]:
                st.metric(label=domain, value=f"{score}/10")
                st.progress(score / 10)
        st.metric(label="Aggregate Score (Original)", value=f"{sum(original_scores.values())/9:.1f}/10")
        st.markdown('</div>', unsafe_allow_html=True)

    if original_response:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Strategic Executive Summary (Original)")
        st.markdown(original_response.split("Improved_Message:")[0])
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if improved_message:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Improved Message")
        st.success(improved_message)
        st.markdown('</div>', unsafe_allow_html=True)

    if improved_scores:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Improved Evaluation Scores")
        cols = st.columns(3)
        for idx, (domain, score) in enumerate(improved_scores.items()):
            with cols[idx % 3]:
                st.metric(label=domain, value=f"{score}/10")
                st.progress(score / 10)
        st.metric(label="Aggregate Score (Improved)", value=f"{sum(improved_scores.values())/9:.1f}/10")
        st.markdown('</div>', unsafe_allow_html=True)

    if improved_response:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Strategic Executive Summary (Improved)")
        st.markdown(improved_response.split("Scores_JSON:")[0])
        st.markdown('</div>', unsafe_allow_html=True)
