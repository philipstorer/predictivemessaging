import streamlit as st
import openai
import pandas as pd
import plotly.graph_objects as go
import json
import re

# Initialize OpenAI client
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

st.title("🎯 Predictive Message Testing Dashboard")
st.caption("Advanced Cognitive-Linguistic Diagnostic and Optimization")

# --- Form Inputs ---
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

# --- Functions ---
def call_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
        max_tokens=3500
    )
    return response.choices[0].message.content

def extract_scores(response_text):
    try:
        json_text = re.search(r'\\{.*\\}', response_text, re.DOTALL).group()
        scores = json.loads(json_text)
        return scores
    except Exception as e:
        print(f"Error parsing scores: {e}")
        return {}

# --- Generate Output ---
if submit_button and original_message:
    with st.spinner('Analyzing your message with deep cognitive-linguistic evaluation...'):
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

Also, output the 9 domain scores at the end in JSON format like this:
{{"Relational Anchoring": 8, "Emotional Reality Validation": 7, "Narrative Integration": 6, "Collaborative Agency Framing": 9, "Value-Embedded Motivation": 8, "Cognitive Effort Reduction": 9, "Temporal Emotional Framing": 7, "Empathic Leadership Positioning": 8, "Affective Modality Matching": 7}}

ORIGINAL MESSAGE:
{original_message}
"""
        original_response = call_gpt(system_prompt_original)

        try:
            improved_message = original_response.split("Suggested Improved Version:")[1].strip().split("\n")[0].replace('> ', '').replace('"', '')
        except:
            improved_message = "(Could not extract improved version.)"

        system_prompt_improved = f"""
You are a senior communication strategist specializing in psycholinguistics.
Evaluate the following IMPROVED MESSAGE according to the Cognitive-Linguistic Deep Analysis Model.
Persona: {persona}
Tone: {tone}

Perform:
- 9 Domain Table (Relational Anchoring, Emotional Reality Validation, Narrative Integration, Collaborative Agency Framing, Value-Embedded Motivation, Cognitive Effort Reduction, Temporal Emotional Framing, Empathic Leadership Positioning, Affective Modality Matching)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary

Also, output the 9 domain scores at the end in JSON format like this:
{{"Relational Anchoring": 8, "Emotional Reality Validation": 7, "Narrative Integration": 6, "Collaborative Agency Framing": 9, "Value-Embedded Motivation": 8, "Cognitive Effort Reduction": 9, "Temporal Emotional Framing": 7, "Empathic Leadership Positioning": 8, "Affective Modality Matching": 7}}

IMPROVED MESSAGE:
{improved_message}
"""
        improved_response = call_gpt(system_prompt_improved)

        st.header("Original Message Evaluation")
        st.write(original_response)

        st.header("Improved Message Evaluation")
        st.write(improved_response)

        try:
            original_scores = extract_scores(original_response)
            improved_scores = extract_scores(improved_response)

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
        except Exception as e:
            st.error(f"Could not parse detailed scores for visualization: {str(e)}")

        st.subheader("Final Improved Message")
        st.success(improved_message)
