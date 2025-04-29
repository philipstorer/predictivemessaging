
import streamlit as st
import openai
import pandas as pd
import plotly.graph_objects as go

# Set your OpenAI API key here
openai.api_key = st.secrets["OPENAI_API_KEY"]

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
   client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": system_prompt}],
    temperature=0.3,
    max_tokens=3500
)

output = response.choices[0].message.content


def extract_scores(response_text):
    lines = response_text.splitlines()
    scores = {}
    capture = False
    for line in lines:
        if "Domain" in line and "Score" in line:
            capture = True
            continue
        if capture:
            if line.strip() == "---":
                break
            parts = line.split("|")
            if len(parts) > 2:
                domain = parts[1].strip()
                try:
                    score = float(parts[2].strip())
                    scores[domain] = score
                except:
                    pass
    return scores

# --- Generate Output ---
if submit_button and original_message:
    with st.spinner('Analyzing your message with deep cognitive-linguistic evaluation...'):
        # Prompt for Original Message
        system_prompt_original = f"""
You are a senior communication strategist specializing in psycholinguistics.
Evaluate the following ORIGINAL MESSAGE according to the Cognitive-Linguistic Deep Analysis Model.
Persona: {persona}
Tone: {tone}

Perform:
- 9 Domain Table (Relational Anchoring, Emotional Reality Validation, etc)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary
- Suggested Improved Version

ORIGINAL MESSAGE:
{original_message}
"""
        original_response = call_gpt(system_prompt_original)

        # Extract Improved Version
        try:
            improved_message = original_response.split("Suggested Improved Version:")[1].strip().split("\n")[0].replace('> ', '').replace('"', '')
        except:
            improved_message = "(Could not extract improved version.)"

        # Prompt for Improved Message Analysis
        system_prompt_improved = f"""
You are a senior communication strategist specializing in psycholinguistics.
Evaluate the following IMPROVED MESSAGE according to the Cognitive-Linguistic Deep Analysis Model.
Persona: {persona}
Tone: {tone}

Perform:
- 9 Domain Table (Relational Anchoring, Emotional Reality Validation, etc)
- Aggregate Cognitive Resonance Score
- Strategic Executive Summary

IMPROVED MESSAGE:
{improved_message}
"""
        improved_response = call_gpt(system_prompt_improved)

        # Layout starts here
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Original Message Evaluation")
            original_scores = extract_scores(original_response)
            original_avg = sum(original_scores.values()) / len(original_scores)
            st.metric("Original Aggregate Score", f"{original_avg:.1f}/10")
            st.write(original_response)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Improved Message Evaluation")
            improved_scores = extract_scores(improved_response)
            improved_avg = sum(improved_scores.values()) / len(improved_scores)
            st.metric("Improved Aggregate Score", f"{improved_avg:.1f}/10")
            st.write(improved_response)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Comparison of Domain Scores")
        comparison_df = pd.DataFrame({
            "Domain": list(original_scores.keys()),
            "Original Score": list(original_scores.values()),
            "Improved Score": [improved_scores.get(d, 0) for d in original_scores.keys()]
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
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Final Improved Message")
        st.success(improved_message)
        st.markdown("</div>", unsafe_allow_html=True)
