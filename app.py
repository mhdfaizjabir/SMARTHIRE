import io
import os

import pandas as pd
import streamlit as st

from updated_extractor import extract_text
from scorer import (
    answer_employer_question,
    generate_interview_questions,
    score_candidate,
)

st.set_page_config(page_title="SmartHire AI", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
    .stApp { background-color: #0f0f1a; color: #e8e8f0; }
    .main-title {
        font-size: 42px; font-weight: 800;
        background: linear-gradient(135deg, #7c6bff, #ff6b9d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #6b6b80; font-size: 14px; margin-bottom: 30px;
        letter-spacing: 2px; text-transform: uppercase;
    }
    .score-green { color: #6bffd8; font-size: 28px; font-weight: 800; }
    .score-yellow { color: #ffd96b; font-size: 28px; font-weight: 800; }
    .score-red { color: #ff6b6b; font-size: 28px; font-weight: 800; }
    .skill-match {
        display: inline-block; background: rgba(107,255,216,0.1); color: #6bffd8;
        border: 1px solid rgba(107,255,216,0.3); border-radius: 20px;
        padding: 3px 10px; font-size: 12px; margin: 2px;
    }
    .skill-miss {
        display: inline-block; background: rgba(255,107,107,0.1); color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.3); border-radius: 20px;
        padding: 3px 10px; font-size: 12px; margin: 2px;
    }
    .section-header {
        font-size: 11px; text-transform: uppercase; letter-spacing: 2px;
        color: #6b6b80; margin-bottom: 8px;
    }
    hr { border-color: #2a2a3a; }
    .streamlit-expanderHeader { background: #1a1a2e !important; border-radius: 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)


def render_explanation(explanation: str, compact: bool = False):
    for line in explanation.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("SUMMARY:"):
            st.markdown(f"📋 **Summary:** {line.replace('SUMMARY:', '').strip()}")
        elif line.startswith("STRENGTHS:"):
            st.success(f"💪 **Strengths:** {line.replace('STRENGTHS:', '').strip()}")
        elif line.startswith("WEAKNESSES:"):
            st.error(f"⚠️ **Weaknesses:** {line.replace('WEAKNESSES:', '').strip()}")
        elif line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
            if "Not Recommended" in verdict:
                label = f"❌ Verdict: {verdict}" if compact else f"### ❌ Verdict: {verdict}"
            elif "Recommended" in verdict:
                label = f"✅ Verdict: {verdict}" if compact else f"### ✅ Verdict: {verdict}"
            else:
                label = f"🔶 Verdict: {verdict}" if compact else f"### 🔶 Verdict: {verdict}"
            st.markdown(f"**{label}**" if compact else label)
        else:
            st.caption(line)



def render_interview_questions(questions_raw: str, compact: bool = False):
    if not isinstance(questions_raw, str):
        return
    current_section = None
    for line in questions_raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("TECHNICAL:"):
            st.markdown("#### 🔧 Technical Questions")
            current_section = "technical"
        elif line.startswith("BEHAVIORAL:"):
            st.markdown("#### 🤝 Behavioral Questions")
            current_section = "behavioral"
        elif line.startswith("GAP:"):
            st.markdown("#### 🔍 Gap Verification Questions")
            current_section = "gap"
        elif line and line[0].isdigit():
            if current_section == "technical":
                st.info(f"💡 {line}")
            elif current_section == "behavioral":
                st.success(f"🤝 {line}")
            elif current_section == "gap":
                st.warning(f"⚠️ {line}")
            else:
                st.write(line)
        elif not compact:
            st.caption(line)


st.markdown('<div class="main-title">🧠 SmartHire AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Enhanced CV Screening · Local AI · Better Grounding</div>',
    unsafe_allow_html=True,
)
st.divider()

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("### 📋 Step 1 — Job Description")
    jd_text = st.text_area(
        "Paste the full job description",
        height=220,
        placeholder="We are looking for a Data Analyst with 2 years experience...",
        label_visibility="collapsed",
    )

with col_right:
    st.markdown("### 📁 Step 2 — Upload CVs")
    uploaded_files = st.file_uploader(
        "Upload CVs",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} CV(s) ready to screen")

st.divider()
st.markdown("### ⚡ Step 3 — Run Screening")
run = st.button("⚡ Run AI Screening", use_container_width=True, type="primary")

if run:
    if not jd_text.strip():
        st.error("⚠️ Please paste a job description first!")
        st.stop()
    if not uploaded_files:
        st.error("⚠️ Please upload at least one CV!")
        st.stop()

    for key in list(st.session_state.keys()):
        if key.startswith("questions_") or key.startswith("interview_") or key == "results":
            del st.session_state[key]

    results = []
    errors = []
    progress = st.progress(0, text="Starting analysis...")
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(int((i / total) * 100), text=f"Analyzing {uploaded_file.name}... ({i+1}/{total})")
        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            cv_text = extract_text(temp_path)
            if not cv_text.strip():
                errors.append(f"Could not read text from {uploaded_file.name}")
                continue
            result = score_candidate(uploaded_file.name, cv_text, jd_text)
            results.append(result)
        except Exception as e:
            errors.append(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    progress.progress(100, text="✅ Analysis complete!")
    for err in errors:
        st.warning(f"⚠️ {err}")
    if not results:
        st.error("No CVs could be processed. Please check your files.")
        st.stop()

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    st.session_state["results"] = results

if "results" in st.session_state:
    results = st.session_state["results"]
    top_score = results[0].get("score", 0)
    if top_score < 30:
        st.warning("⚠️ No strong matches found. All candidates scored below 30/100.")

    st.divider()
    res_col1, res_col2 = st.columns([3, 1])
    with res_col1:
        st.markdown(f"### 🏆 Ranking Results — {len(results)} Candidates Screened")
    with res_col2:
        export_data = []
        for i, c in enumerate(results):
            export_data.append(
                {
                    "Rank": i + 1,
                    "Name": c.get("name", ""),
                    "Score": c.get("score", 0),
                    "Blended Similarity %": c.get("similarity", 0),
                    "Lexical Similarity %": c.get("lexical_similarity", 0),
                    "Semantic Similarity %": c.get("semantic_similarity", 0),
                    "Matched Skills": ", ".join(c.get("matched_skills", [])),
                    "Missing Skills": ", ".join(c.get("missing_skills", [])),
                    "Summary": c.get("summary", ""),
                    "Explanation": c.get("explanation", ""),
                }
            )
        df_export = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export CSV",
            data=csv_buffer.getvalue(),
            file_name="smarthire_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    for i, candidate in enumerate(results):
        rank = i + 1
        score = candidate.get("score", 0)
        name = candidate.get("name", candidate.get("filename", "Unknown"))
        similarity = candidate.get("similarity", 0)
        lexical = candidate.get("lexical_similarity", 0)
        semantic = candidate.get("semantic_similarity", 0)
        matched = candidate.get("matched_skills", [])
        missing = candidate.get("missing_skills", [])
        scores = candidate.get("scores", {})

        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        score_class = "score-green" if score >= 70 else "score-yellow" if score >= 50 else "score-red"
        score_emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"

        with st.expander(f"{medal}  {name}  ·  {score_emoji} {score}/100  ·  {similarity}% blended similarity"):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown('<div class="section-header">Overall Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{score_class}">{score}/100</div>', unsafe_allow_html=True)
                st.caption(f"Blended similarity: {similarity}%")
                st.caption(f"Lexical TF-IDF: {lexical}%")
                st.caption(f"Semantic retrieval: {semantic}%")
            with c2:
                st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
                st.progress(scores.get("skills", 0) / 100, text=f"Skills: {scores.get('skills', 0)}/100")
                st.progress(scores.get("experience", 0) / 100, text=f"Experience: {scores.get('experience', 0)}/100")
                st.progress(scores.get("education", 0) / 100, text=f"Education: {scores.get('education', 0)}/100")
                st.progress(scores.get("projects", 0) / 100, text=f"Projects: {scores.get('projects', 0)}/100")
            with c3:
                st.markdown('<div class="section-header">Matched Skills</div>', unsafe_allow_html=True)
                if matched:
                    st.markdown(" ".join([f'<span class="skill-match">✓ {s}</span>' for s in matched]), unsafe_allow_html=True)
                else:
                    st.caption("No skills matched")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Missing Skills</div>', unsafe_allow_html=True)
                if missing:
                    st.markdown(" ".join([f'<span class="skill-miss">✗ {s}</span>' for s in missing]), unsafe_allow_html=True)
                else:
                    st.caption("None — perfect skill match! 🎉")

            st.divider()
            if candidate.get("is_suspicious"):
                st.warning(f"⚠️ Fraud Alert: Skills not backed by work evidence. Trust score: {candidate.get('trust_score')}/1.0")

            st.markdown('<div class="section-header">📎 Retrieved Evidence</div>', unsafe_allow_html=True)
            evidence_chunks = candidate.get("evidence_chunks", [])
            if evidence_chunks:
                for chunk in evidence_chunks[:4]:
                    st.info(f"**{chunk.get('section', 'other').title()}**\n\n{chunk.get('text', '')[:400]}")
            else:
                st.caption("No evidence chunks available.")

            st.divider()
            st.markdown('<div class="section-header">🤖 AI Evaluation</div>', unsafe_allow_html=True)
            render_explanation(candidate.get("explanation", ""))

            st.divider()
            st.markdown('<div class="section-header">🎯 Interview Preparation</div>', unsafe_allow_html=True)
            btn_key = f"interview_{i}_{name}"
            if st.button(f"🎤 Generate Interview Questions for {name}", key=btn_key):
                with st.spinner(f"Generating tailored interview questions for {name}..."):
                    questions_raw = generate_interview_questions(
                        candidate.get("cv_profile", {}),
                        candidate.get("jd_profile", {}),
                        candidate.get("result_raw", {}),
                    )
                    st.session_state[f"questions_{i}"] = questions_raw
            if f"questions_{i}" in st.session_state:
                render_interview_questions(st.session_state[f"questions_{i}"])

    st.divider()
    st.markdown("### 💬 Employer Chat Assistant")
    st.caption("Ask questions about ranked candidates. Answers are grounded in retrieved CV evidence.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_question = st.text_input("Ask something about the candidates", placeholder="Who is the best candidate and why?")
    if st.button("Ask Chatbot", key="ask_chatbot_btn"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                answer = answer_employer_question(user_question, results, jd_text)
                st.session_state["chat_history"].append(("You", user_question))
                st.session_state["chat_history"].append(("Assistant", answer))

    for role, message in st.session_state["chat_history"]:
        st.markdown(f"**🧑 Employer:** {message}" if role == "You" else f"**🤖 Assistant:** {message}")
