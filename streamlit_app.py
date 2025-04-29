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

# Style
st.markdown("""
<style>
.section-title {
    font-size:22px;
    font-weight:bold;
    margin-top:20px;
}
.metric-card {
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Predictive Message Testing Dashboard")
st.caption("Advanced Cognitive-Linguistic Diagnostic and Optimization")

# Layout
left, right = st.columns([1, 2])

with left:
    st.header("Message Input")
    with st.form("input_form"):
        original_message = st.text_area("Enter Original Message:", height=200)
        persona = st.selectbox("Select Target Persona:", [
            "Women 45-65, Caregiving Role",
            "Female Millennials, Career-Focused",
            "Female Patients, Chronic Illness Management",
            "Female Healthcare Providers, Clinical Setting"
        ])
        tone = st.selectbox("Desired Tone:", ["Empathetic", "Clinical", "Inspirational", "Direct"], index=0)
        submit_button = st.form_submit_button("Analyze Message")

with right:
    evaluation_placeholder = st.empty()

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

if submit_button and original_message:
    with evaluation_placeholder.container():
        st.info("Starting cognitive-linguistic analysis...")
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
            st.write(message)
            time.sleep(0.8)

    original_length = len(original_message)

    # Prompt
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

Constraints for the Improved Version:
- Preserve all explicit factual claims and value propositions.
- Maintain the original communication intent and fundamental ideas.
- Improve emotional resonance, tone, flow, and readability.
- Allow moderate rewording if needed for clarity.
- Stay within ±15% of the original character count ({original_length} characters).

At the end, clearly output:
Improved_Message: (improved message text)

Also output:
Scores_JSON: (9 cognitive domain scores)
"""
    original_response = call_gpt(system_prompt_original)

    improved_message = extract_improved_message(original_response)
    original_scores = extract_json_block(original_response, "Scores_JSON")

    improved_response = ""
    improved_scores = {}

    if improved_message:
        system_prompt_improved = f"""
You are a senior communication strategist specializing in psycholinguistics.

Evaluate the following IMPROVED MESSAGE according to the Cognitive-Linguistic Deep Analysis Model.
Persona: {persona}
Tone: {tone}

Perform:
- Output a Markdown table:
    | Domain | Score (0-10) | Diagnostic Insight | Strategic Impact |
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary

At the end, clearly output:
Scores_JSON: (9 cognitive domain scores)
"""
        improved_response = call_gpt(system_prompt_improved)
        improved_scores = extract_json_block(improved_response, "Scores_JSON")

    evaluation_placeholder.empty()

    st.divider()

    # Results
    st.subheader("Original Message")
    st.write(original_message)

    st.subheader("Evaluation Criteria Explanations")
    st.markdown("""
- **Relational Anchoring**: Connection to others.
- **Emotional Reality Validation**: Acknowledges feelings.
- **Narrative Integration**: Storytelling strength.
- **Collaborative Agency Framing**: Encourages participation.
- **Value-Embedded Motivation**: Ties action to values.
- **Cognitive Effort Reduction**: Easy to understand.
- **Temporal Emotional Framing**: Sense of urgency or timeliness.
- **Empathic Leadership Positioning**: Confidence and compassion.
- **Affective Modality Matching**: Matches emotional style.
""")

    if original_scores:
        st.subheader("Original Message Scores")
        cols = st.columns(3)
        for idx, (domain, score) in enumerate(original_scores.items()):
            with cols[idx % 3]:
                st.markdown(f"<div class='metric-card'><strong>{domain}</strong><br>", unsafe_allow_html=True)
                st.progress(score / 10)
                st.markdown("</div>", unsafe_allow_html=True)

        st.metric(label="Aggregate Score (Original)", value=f"{sum(original_scores.values())/9:.1f}/10")

    st.subheader("Strategic Executive Summary (Original)")
    st.markdown(original_response.split("Improved_Message:")[0].split("| Strategic Impact |")[1].split("Aggregate Cognitive Resonance Score")[1].strip())

    st.divider()

    if improved_message:
        st.subheader("Improved Message")
        st.success(improved_message)

    if improved_scores:
        st.subheader("Improved Message Scores")
        cols = st.columns(3)
        for idx, (domain, score) in enumerate(improved_scores.items()):
            with cols[idx % 3]:
                st.markdown(f"<div class='metric-card'><strong>{domain}</strong><br>", unsafe_allow_html=True)
                st.progress(score / 10)
                st.markdown("</div>", unsafe_allow_html=True)

        st.metric(label="Aggregate Score (Improved)", value=f"{sum(improved_scores.values())/9:.1f}/10")

    if improved_response:
        st.subheader("Strategic Executive Summary (Improved)")
        st.markdown(improved_response.split("Scores_JSON:")[0].split("| Strategic Impact |")[1].split("Aggregate Cognitive Resonance Score")[1].strip())
