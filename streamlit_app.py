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
        match = re.search(r'Improved_Message:\s*(.*)', response_text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = extracted.split('Scores_JSON:')[0].strip()
            return extracted
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
- 9 Domain Table (Relational Anchoring, Emotional Reality Validation, Narrative Integration, Collaborative Agency Framing, Value-Embedded Motivation, Cognitive Effort Reduction, Temporal Emotional Framing, Empathic Leadership Positioning, Affective Modality Matching)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary
- Suggested Improved Version

Constraints for the Improved Version:
- Light, subtle improvements.
- Maintain same fundamental ideas and meaning.
- Stay within ±15% of the original character count ({original_length} characters).

Clearly output:
Improved_Message: 
(Write the improved message here)

Scores_JSON: 
{{"Relational Anchoring": 8, "Emotional Reality Validation": 7, "Narrative Integration": 6, "Collaborative Agency Framing": 9, "Value-Embedded Motivation": 8, "Cognitive Effort Reduction": 9, "Temporal Emotional Framing": 7, "Empathic Leadership Positioning": 8, "Affective Modality Matching": 7}}
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
- 9 Domain Table
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary

Clearly output:
Scores_JSON: 
{{"Relational Anchoring": 8, "Emotional Reality Validation": 7, "Narrative Integration": 6, "Collaborative Agency Framing": 9, "Value-Embedded Motivation": 8, "Cognitive Effort Reduction": 9, "Temporal Emotional Framing": 7, "Empathic Leadership Positioning": 8, "Affective Modality Matching": 7}}
"""
            improved_response = call_gpt(system_prompt_improved)
            improved_scores = extract_json_block(improved_response, "Scores_JSON")

    st.header("Original Message Evaluation")
    if original_response:
        st.markdown(original_response.split("Scores_JSON:")[0])

    if improved_message:
        st.header("Improved Message Evaluation")
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

        st.subheader("Final Improved Message")
        st.success(improved_message)
    else:
        st.error("No improved message could be extracted. Please refine input.")
