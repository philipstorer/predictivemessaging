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
.section-title {
    font-size:22px;
    font-weight:bold;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

st.title("Predictive Message Testing Dashboard")
st.caption("Advanced Cognitive-Linguistic Diagnostic and Optimization")

spinner_placeholder = st.empty()
thinking_placeholder = st.empty()

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
            json_obj = json.loads(match.group(1))
            return json_obj
        else:
            return None
    except Exception as e:
        print(f"Error parsing {label}: {e}")
        return None

def extract_improved_message(response_text):
    try:
        match = re.search(r'Improved_Message:\s*"(.*?)"', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r'Improved_Message:\s*(.*)', response_text)
            if match:
                return match.group(1).strip()
            else:
                return None
    except Exception as e:
        print(f"Error extracting improved message: {e}")
        return None

if submit_button and original_message:
    spinner_placeholder = st.spinner('Starting cognitive-linguistic analysis...')
    thinking_area = st.empty()

    with spinner_placeholder:
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
            thinking_area.info(message)
            time.sleep(1.2)
            thinking_area.empty()

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
- It must maintain the same fundamental idea, purpose, and meaning as the original message.
- It must stay within ±15% of the original character count ({original_length} characters).
- It should improve tone, readability, emotional resonance, and strategic impact, without altering the core communication intent.

Output formatting:
- After the evaluation, output ONLY the improved message separately, clearly labeled like this:
Improved_Message: "(Your improved message here)"

- Then output the 9 domain scores clearly in JSON format like this:
Scores_JSON: {{"Relational Anchoring": 8, "Emotional Reality Validation": 7, "Narrative Integration": 6, "Collaborative Agency Framing": 9, "Value-Embedded Motivation": 8, "Cognitive Effort Reduction": 9, "Temporal Emotional Framing": 7, "Empathic Leadership Positioning": 8, "Affective Modality Matching": 7}}
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
- 9 Domain Table (Relational Anchoring, Emotional Reality Validation, Narrative Integration, Collaborative Agency Framing, Value-Embedded Motivation, Cognitive Effort Reduction, Temporal Emotional Framing, Empathic Leadership Positioning, Affective Modality Matching)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary

Output formatting:
- After the evaluation, output the 9 domain scores clearly in JSON format like this:
Scores_JSON: {{"Relational Anchoring": 8, "Emotional Reality Validation": 7, "Narrative Integration": 6, "Collaborative Agency Framing": 9, "Value-Embedded Motivation": 8, "Cognitive Effort Reduction": 9, "Temporal Emotional Framing": 7, "Empathic Leadership Positioning": 8, "Affective Modality Matching": 7}}
"""
        improved_response = call_gpt(system_prompt_improved)
        improved_scores = extract_json_block(improved_response, "Scores_JSON")

    st.markdown('<div class="section-title">Original Message Evaluation</div>', unsafe_allow_html=True)
    st.markdown(original_response.split("Scores_JSON:")[0])

    if improved_message:
        st.markdown('<div class="section-title">Improved Message Evaluation</div>', unsafe_allow_html=True)
        st.markdown(improved_response.split("Scores_JSON:")[0])

        if original_scores and improved_scores:
            st.markdown('<div class="section-title">Comparison of Domain Scores</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])

            with col2:
                comparison_df = pd.DataFrame({
                    "Domain": original_scores.keys(),
                    "Original Score": original_scores.values(),
                    "Improved Score": [improved_scores.get(domain, 0) for domain in original_scores.keys()]
                })

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

            with col1:
                st.metric(label="Aggregate Score (Original)", value=f"{sum(original_scores.values())/9:.1f}/10")
                st.metric(label="Aggregate Score (Improved)", value=f"{sum(improved_scores.values())/9:.1f}/10")

        st.markdown('<div class="section-title">Final Improved Message</div>', unsafe_allow_html=True)
        st.success(improved_message)
    else:
        st.error("No improved message could be extracted. Please refine input.")
