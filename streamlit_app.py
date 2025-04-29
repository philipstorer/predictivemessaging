import streamlit as st
import openai
import pandas as pd
import plotly.graph_objects as go
import json
import re
import time

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Predictive Message Testing Dashboard", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
.card {
    background-color: #f9f9f9;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("Predictive Message Testing Dashboard")
st.caption("Advanced Cognitive-Linguistic Diagnostic and Optimization")

with st.form("message_form"):
    st.header("Message Input")
    original_message = st.text_area("Enter Original Message:", height=200)
    persona = st.selectbox("Select Target Persona:", [
        "Women 45-65, Caregiving Role",
        "Female Millennials, Career-Focused",
        "Female Patients, Chronic Illness Management",
        "Female Healthcare Providers, Clinical Setting"
    ])
    tone = st.selectbox("Desired Tone:", ["Empathetic", "Clinical", "Inspirational", "Direct"], index=0)
    submit_button = st.form_submit_button("Analyze Message")

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

    with st.spinner('Starting cognitive-linguistic analysis...'):
        for message in spinner_messages:
            with st.empty():
                st.info(message)
                time.sleep(1.5)

        original_length = len(original_message)

        system_prompt_original = f"""
You are a senior communication strategist specializing in psycholinguistics.

Evaluate the following ORIGINAL MESSAGE according to the Cognitive-Linguistic Deep Analysis Model.
Persona: {persona}
Tone: {tone}

Perform:
- Output a clear Markdown table showing:
    | Domain | Score (0-10) | Diagnostic Insight | Strategic Impact |
    |--------|--------------|--------------------|------------------|
    (One row per domain)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary
- THEN suggest an improved version.

Constraints for the Improved Version:
- Preserve all explicit factual claims, product benefits, and value propositions like but not limited to benefits such as flexible dosing and quick onset of action exactly. This is a non-negotiable. Check it to make sure this is preserved. 
- Maintain the original communication intent and fundamental ideas.
- Improve emotional resonance, tone, flow, and overall readability.
- Allow moderate rewording and restructuring if needed for clarity and engagement.
- Stay within ±15% of the original character count ({original_length} characters).

At the end, clearly output:
Improved_Message: (the improved message)

Also output:
Scores_JSON: (the 9 domain scores in JSON format)
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
- Output a clear Markdown table showing:
    | Domain | Score (0-10) | Diagnostic Insight | Strategic Impact |
    |--------|--------------|--------------------|------------------|
    (One row per domain)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary

At the end, clearly output:
Scores_JSON: (the 9 domain scores in JSON format)
"""
            improved_response = call_gpt(system_prompt_improved)
            improved_scores = extract_json_block(improved_response, "Scores_JSON")

    st.header("Original Message Evaluation")
    if original_response:
        st.markdown(original_response.split("Improved_Message:")[0])

    if improved_message:
        st.header("Improved Message Evaluation")
        st.success(improved_message)

        if improved_response:
            st.markdown(improved_response.split("Scores_JSON:")[0])

        if original_scores and improved_scores:
            comparison_df = pd.DataFrame({
                "Domain": original_scores.keys(),
                "Original Score": original_scores.values(),
                "Improved Score": [improved_scores.get(domain, 0) for domain in original_scores.keys()]
            })

            st.subheader("Comparison of Domain Scores")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=comparison_df["Domain"],
                x=comparison_df["Original Score"],
                name='Original Score',
                orientation='h'
            ))
            fig.add_trace(go.Bar(
                y=comparison_df["Domain"],
                x=comparison_df["Improved Score"],
                name='Improved Score',
                orientation='h'
            ))
            fig.update_layout(barmode='group', height=600)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No improved message could be extracted. Please refine input.")
