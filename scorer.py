import json
import re
from typing import Any, Dict, List, Optional

import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from updated_extractor import build_cv_chunks, normalize_whitespace, split_into_sections
from retrieval import retrieve_top_k, semantic_similarity_from_chunks


OLLAMA_MODEL = "gemma3:1b"


DEFAULT_JD_PROFILE = {
    "required_skills": [],
    "job_role": "",
    "years_required": 0,
    "education_required": "none",
}

DEFAULT_CV_PROFILE = {
    "candidate_name": "Unknown",
    "skills": [],
    "job_roles_held": [],
    "years_of_experience": 0,
    "education_level": "none",
    "has_projects": False,
    "project_descriptions": [],
}



def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default



def normalize_skill_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    cleaned = []
    seen = set()
    for item in items:
        value = normalize_whitespace(str(item)).lower()
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned



def parse_first_json_object(raw_text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not raw_text:
        return dict(fallback)

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return dict(fallback)

    try:
        parsed = json.loads(match.group())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return dict(fallback)

    return dict(fallback)



def parse_first_json_array(raw_text: str, fallback: Optional[List[Any]] = None) -> List[Any]:
    if fallback is None:
        fallback = []
    if not raw_text:
        return list(fallback)

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return list(fallback)

    try:
        parsed = json.loads(match.group())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return list(fallback)

    return list(fallback)



def ollama_chat(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()



def tfidf_similarity(jd_text: str, cv_text: str) -> float:
    jd_text = normalize_whitespace(jd_text)
    cv_text = normalize_whitespace(cv_text)
    if not jd_text or not cv_text:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([jd_text, cv_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(score) * 100, 2)



def build_retrieval_query(jd_profile: Dict[str, Any], jd_text: str) -> str:
    skills = ", ".join(jd_profile.get("required_skills", []))
    role = jd_profile.get("job_role", "")
    years = jd_profile.get("years_required", 0)
    education = jd_profile.get("education_required", "")
    return normalize_whitespace(
        f"Target role: {role}. Required skills: {skills}. Years: {years}. Education: {education}. "
        f"Job description: {jd_text[:1200]}"
    )



def select_relevant_cv_context(cv_text: str, jd_profile: Dict[str, Any], jd_text: str, top_k: int = 6):
    cv_chunks = build_cv_chunks(cv_text)
    query = build_retrieval_query(jd_profile, jd_text)
    top_chunks = retrieve_top_k(query, cv_chunks, k=top_k)
    context = "\n\n".join(
        f"[{chunk.get('section', 'other').upper()}]\n{chunk.get('text', '')}"
        for chunk in top_chunks
    )
    return cv_chunks, top_chunks, context



def extract_requirements(jd_text: str) -> Dict[str, Any]:
    prompt = f"""Read this job description carefully.
Extract SPECIFIC skills, not broad terms.
Return ONLY valid JSON in this exact schema:
{{
  "required_skills": ["python", "sql", "power bi"],
  "job_role": "exact job title",
  "years_required": 2,
  "education_required": "bachelor"
}}

Rules:
- skills must be specific technologies, tools, or methods
- years_required must be a number
- education_required must be one of: none, diploma, bachelor, master, phd
- if missing, use empty string or 0

Job Description:
{jd_text}
"""
    try:
        raw = ollama_chat(prompt)
        profile = parse_first_json_object(raw, DEFAULT_JD_PROFILE)
    except Exception as e:
        profile = dict(DEFAULT_JD_PROFILE)
        profile["_error"] = str(e)

    profile["required_skills"] = normalize_skill_list(profile.get("required_skills", []))
    profile["job_role"] = normalize_whitespace(str(profile.get("job_role", "")))
    profile["years_required"] = safe_int(profile.get("years_required", 0), 0)
    profile["education_required"] = normalize_whitespace(
        str(profile.get("education_required", "none"))
    ).lower() or "none"
    return profile



def extract_cv_profile(cv_text: str, jd_profile: Dict[str, Any], jd_text: str) -> Dict[str, Any]:
    sections = split_into_sections(cv_text)
    cv_chunks, top_chunks, retrieved_context = select_relevant_cv_context(cv_text, jd_profile, jd_text)

    strong_section_context = "\n\n".join(
        f"[{name.upper()}]\n{content}"
        for name, content in sections.items()
        if name in {"skills", "experience", "projects", "education", "summary"}
    )

    evidence_context = normalize_whitespace(f"{strong_section_context}\n\n{retrieved_context}")[:9000]

    prompt = f"""Read this CV evidence carefully and extract a structured candidate profile.
Return ONLY valid JSON in this exact schema:
{{
  "candidate_name": "full name or Unknown",
  "skills": ["python", "sql", "power bi"],
  "job_roles_held": ["Data Analyst", "Business Analyst"],
  "years_of_experience": 3,
  "education_level": "bachelor",
  "has_projects": true,
  "project_descriptions": ["built dashboard for sales reporting"]
}}

Rules:
- skills may include explicit skills and very strongly supported tools from project evidence
- job roles must be exact or near-exact roles mentioned in the CV evidence
- years_of_experience must be numeric
- education_level must be one of: none, diploma, bachelor, master, phd
- do not invent credentials not supported by the evidence

Target Job Role:
{jd_profile.get('job_role', '')}
Required Skills:
{jd_profile.get('required_skills', [])}

Retrieved CV Evidence:
{evidence_context}
"""

    try:
        raw = ollama_chat(prompt)
        profile = parse_first_json_object(raw, DEFAULT_CV_PROFILE)
    except Exception as e:
        profile = dict(DEFAULT_CV_PROFILE)
        profile["_error"] = str(e)

    profile["candidate_name"] = normalize_whitespace(str(profile.get("candidate_name", "Unknown"))) or "Unknown"
    profile["skills"] = normalize_skill_list(profile.get("skills", []))
    profile["job_roles_held"] = [normalize_whitespace(str(r)) for r in profile.get("job_roles_held", []) if normalize_whitespace(str(r))]
    profile["years_of_experience"] = safe_int(profile.get("years_of_experience", 0), 0)
    profile["education_level"] = normalize_whitespace(str(profile.get("education_level", "none"))).lower() or "none"
    profile["has_projects"] = bool(profile.get("has_projects", False) or profile.get("project_descriptions"))
    profile["project_descriptions"] = [normalize_whitespace(str(p)) for p in profile.get("project_descriptions", []) if normalize_whitespace(str(p))]
    profile["retrieved_chunks"] = top_chunks
    profile["cv_chunks"] = cv_chunks
    profile["retrieved_context"] = evidence_context
    return profile



def cross_validate_skills(matched_skills: List[str], cv_text: str):
    cv_lower = cv_text.lower()

    skills_section = ""
    evidence_section = ""

    lines = cv_lower.split("\n")
    current_section = "other"

    for line in lines:
        if any(word in line for word in ["skill", "technical", "competenc"]):
            current_section = "skills"
        elif any(word in line for word in ["experience", "work", "employment", "project", "responsibilities"]):
            current_section = "evidence"

        if current_section == "skills":
            skills_section += line + " "
        elif current_section == "evidence":
            evidence_section += line + " "

    if not evidence_section.strip():
        evidence_section = cv_lower

    total_trust = 0.0
    skill_trust_details = {}

    for skill in matched_skills:
        in_evidence = skill in evidence_section
        in_skills_section = skill in skills_section

        if in_evidence:
            trust = 1.0
        elif in_skills_section:
            trust = 0.5
        else:
            trust = 0.3

        skill_trust_details[skill] = trust
        total_trust += trust

    if not matched_skills:
        return 0.0, skill_trust_details

    avg_trust = total_trust / len(matched_skills)
    return round(avg_trust, 2), skill_trust_details



def skills_match(required_skill: str, candidate_skills: List[str], cv_text_lower: str = "") -> bool:
    if any(required_skill in c or c in required_skill for c in candidate_skills):
        return True

    req_words = [w for w in required_skill.split() if len(w) > 3]
    for c in candidate_skills:
        c_words = [w for w in c.split() if len(w) > 3]
        if any(rw in c for rw in req_words):
            return True
        if any(cw in required_skill for cw in c_words):
            return True

    if cv_text_lower and len(required_skill.split()) >= 2:
        if all(word in cv_text_lower for word in req_words):
            return True

    return False



def compare_and_score(
    jd_profile: Dict[str, Any],
    cv_profile: Dict[str, Any],
    lexical_similarity: float,
    semantic_similarity: float,
    cv_text: str = "",
    jd_text: str = "",
) -> Dict[str, Any]:
    cv_text_lower = cv_text.lower()

    required = [s.lower().strip() for s in jd_profile.get("required_skills", [])]
    candidate = [s.lower().strip() for s in cv_profile.get("skills", [])]

    matched_skills = [s for s in required if skills_match(s, candidate, cv_text_lower)]
    missing_skills = [s for s in required if not skills_match(s, candidate, cv_text_lower)]

    trust_score, skill_trust_details = cross_validate_skills(matched_skills, cv_text)

    lexical_component = round((lexical_similarity / 100) * 10)
    semantic_component = round((semantic_similarity / 100) * 10)
    skills_score = lexical_component + semantic_component

    if required:
        direct_match_ratio = len(matched_skills) / len(required)
        direct_bonus = round(direct_match_ratio * 20 * trust_score)
        skills_score = min(skills_score + direct_bonus, 40)
    else:
        skills_score = min(skills_score + 5, 40)

    is_suspicious = trust_score < 0.6 and len(matched_skills) >= 3

    required_role = jd_profile.get("job_role", "").lower().strip()
    held_roles = [r.lower().strip() for r in cv_profile.get("job_roles_held", [])]
    years_required = safe_int(jd_profile.get("years_required", 0), 0)
    cv_years = safe_int(cv_profile.get("years_of_experience", 0), 0)

    role_relevant = False
    if required_role and held_roles:
        required_words = [w for w in required_role.split() if len(w) > 3]
        for held in held_roles:
            if any(word in held for word in required_words):
                role_relevant = True
                break
            held_words = [w for w in held.split() if len(w) > 3]
            if any(word in required_role for word in held_words):
                role_relevant = True
                break

    if not role_relevant and len(matched_skills) >= 2:
        jd_lower = jd_text.lower()
        for held in held_roles:
            for word in held.split():
                if len(word) > 4 and word in jd_lower:
                    role_relevant = True
                    break
            if role_relevant:
                break

    exp_score = 20 if role_relevant else 4
    if years_required == 0:
        exp_score += 10
    elif cv_years >= years_required:
        exp_score += 10
    elif cv_years >= max(0, years_required - 1):
        exp_score += 6
    elif cv_years > 0:
        exp_score += 3
    exp_score = min(exp_score, 30)

    edu_required = jd_profile.get("education_required", "none").lower()
    edu_candidate = cv_profile.get("education_level", "none").lower()

    edu_levels = {
        "phd": 4, "doctorate": 4,
        "master": 3, "msc": 3, "mba": 3,
        "bachelor": 2, "bsc": 2, "degree": 2,
        "diploma": 1, "certificate": 1,
        "none": 0,
    }

    required_level = edu_levels.get(edu_required, 0)
    candidate_level = edu_levels.get(edu_candidate, 0)

    if candidate_level == 0 and cv_text_lower:
        for edu_keyword, level in edu_levels.items():
            if edu_keyword in cv_text_lower and level > candidate_level:
                candidate_level = level

    if required_level == 0:
        edu_score = 10
    elif candidate_level >= required_level:
        edu_score = 15
    elif candidate_level == required_level - 1:
        edu_score = 8
    else:
        edu_score = 3

    has_projects = cv_profile.get("has_projects", False)
    project_text = " ".join(cv_profile.get("project_descriptions", [])).lower()
    if not has_projects and any(w in cv_text_lower for w in ["project", "built", "developed", "designed", "constructed"]):
        has_projects = True

    proj_score = 0
    if has_projects:
        proj_score += 7
        relevant_count = sum(1 for s in matched_skills if s in project_text or s in cv_text_lower)
        if relevant_count >= 2:
            proj_score += 8
        elif relevant_count >= 1:
            proj_score += 4
    proj_score = min(proj_score, 15)

    total = skills_score + exp_score + edu_score + proj_score
    blended_similarity = round(0.45 * lexical_similarity + 0.55 * semantic_similarity, 2)

    return {
        "total": total,
        "skills_score": skills_score,
        "exp_score": exp_score,
        "edu_score": edu_score,
        "proj_score": proj_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "role_relevant": role_relevant,
        "cv_years": cv_years,
        "lexical_similarity": lexical_similarity,
        "semantic_similarity": semantic_similarity,
        "blended_similarity": blended_similarity,
        "trust_score": trust_score,
        "is_suspicious": is_suspicious,
        "skill_trust_details": skill_trust_details,
    }



def get_explanation(cv_profile: Dict[str, Any], jd_profile: Dict[str, Any], result: Dict[str, Any]) -> str:
    matched_count = len(result.get("matched_skills", []))
    lexical = result.get("lexical_similarity", 0)
    semantic = result.get("semantic_similarity", 0)
    role_relevant = result.get("role_relevant", False)
    total_score = result.get("total", 0)

    evidence_lines = []
    for chunk in cv_profile.get("retrieved_chunks", [])[:4]:
        section = chunk.get("section", "other").upper()
        text = chunk.get("text", "")[:300]
        evidence_lines.append(f"[{section}] {text}")
    evidence_block = "\n".join(evidence_lines)

    if matched_count == 0 and semantic < 15 and not role_relevant:
        return (
            "SUMMARY: The candidate's background does not align with the requirements of this role.\n"
            "STRENGTHS: There are no major role-relevant strengths supported by the retrieved evidence.\n"
            "WEAKNESSES: The evidence does not show the required technical skills or directly relevant role experience.\n"
            "VERDICT: Not Recommended"
        )

    prompt = f"""You are a strict recruiter writing a structured candidate evaluation.
Use ONLY the retrieved evidence below.
Do not invent details beyond the evidence.

Job role needed: {jd_profile.get('job_role')}
Required skills: {jd_profile.get('required_skills')}
Required years: {jd_profile.get('years_required')}
Required education: {jd_profile.get('education_required')}

Candidate: {cv_profile.get('candidate_name')}
Extracted skills: {cv_profile.get('skills')}
Roles held: {cv_profile.get('job_roles_held')}
Years experience: {cv_profile.get('years_of_experience')}
Matched skills: {result.get('matched_skills')}
Missing skills: {result.get('missing_skills')}
Relevant experience: {result.get('role_relevant')}
Lexical similarity: {lexical}%
Semantic similarity: {semantic}%
Trust score: {result.get('trust_score')} / 1.0
Total score: {result.get('total')}/100

Retrieved Evidence:
{evidence_block}

Write EXACTLY in this format:
SUMMARY: one sentence overall verdict for this role only.
STRENGTHS: only role-relevant strengths supported by evidence.
WEAKNESSES: missing skills, missing experience, or unsupported claims.
VERDICT: Recommended / Potential / Not Recommended
"""

    try:
        raw = ollama_chat(prompt)
        if all(tag in raw for tag in ["SUMMARY:", "STRENGTHS:", "WEAKNESSES:", "VERDICT:"]):
            return raw.strip()
    except Exception:
        pass

    verdict = "Recommended" if total_score >= 70 else "Potential" if total_score >= 50 else "Not Recommended"
    return (
        f"SUMMARY: The candidate shows {'good' if total_score >= 70 else 'partial' if total_score >= 50 else 'limited'} alignment with the role based on retrieved CV evidence.\n"
        f"STRENGTHS: {('Matches skills such as ' + ', '.join(result.get('matched_skills', [])[:3])) if matched_count > 0 else 'There are no major role-relevant strengths clearly supported.'}\n"
        f"WEAKNESSES: Missing {', '.join(result.get('missing_skills', [])[:3]) if result.get('missing_skills') else 'important required skills'} or stronger supporting evidence.\n"
        f"VERDICT: {verdict}"
    )



def generate_interview_questions(cv_profile: Dict[str, Any], jd_profile: Dict[str, Any], result: Dict[str, Any]) -> str:
    projects = cv_profile.get("project_descriptions", [])
    project_text = projects[0] if projects else "no specific project mentioned"
    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])
    evidence_lines = [chunk.get("text", "")[:240] for chunk in cv_profile.get("retrieved_chunks", [])[:3]]

    prompt = f"""You are an expert HR interviewer.
Generate specific interview questions using ONLY this candidate evidence.

Job Role: {jd_profile.get('job_role')}
Required Skills: {jd_profile.get('required_skills')}
Candidate: {cv_profile.get('candidate_name')}
Their Skills: {cv_profile.get('skills')}
Roles They Held: {cv_profile.get('job_roles_held')}
Years Experience: {cv_profile.get('years_of_experience')}
Projects: {cv_profile.get('project_descriptions')}
Matched Skills: {matched}
Missing Skills: {missing}
Retrieved Evidence: {evidence_lines}

Write EXACTLY in this format:
TECHNICAL:
1. question
2. question
3. question
BEHAVIORAL:
1. question
2. question
GAP:
1. question
2. question
"""

    try:
        raw = ollama_chat(prompt)
        if "TECHNICAL:" in raw and "BEHAVIORAL:" in raw and "GAP:" in raw:
            return raw.strip()
    except Exception as e:
        return f"TECHNICAL:\n1. Error generating questions: {str(e)}\nBEHAVIORAL:\n1. Please try again.\nGAP:\n1. Please try again."

    return (
        "TECHNICAL:\n"
        f"1. Walk me through your experience with {matched[0] if matched else 'your strongest technical skill'}.\n"
        f"2. How did you apply {matched[1] if len(matched) > 1 else matched[0] if matched else 'your tools'} in a real task or project?\n"
        f"3. Explain the design decisions behind this project: {project_text}.\n"
        "BEHAVIORAL:\n"
        f"1. Tell me about a challenge you faced in a role related to {jd_profile.get('job_role')}.\n"
        "2. Describe a time you had to learn a new tool quickly.\n"
        "GAP:\n"
        f"1. How would you close your gap in {missing[0] if missing else 'one required skill'}?\n"
        f"2. What is your plan to become stronger in {missing[1] if len(missing) > 1 else missing[0] if missing else 'another missing area'}?"
    )



def answer_employer_question(question: str, results: List[Dict[str, Any]], jd_text: str) -> str:
    if not question.strip():
        return "Please enter a question."

    evidence_pool = []
    candidate_summary = []
    for candidate in results:
        candidate_summary.append(
            {
                "name": candidate.get("name", candidate.get("filename", "Unknown")),
                "score": candidate.get("score", 0),
                "blended_similarity": candidate.get("similarity", 0),
                "matched_skills": candidate.get("matched_skills", []),
                "missing_skills": candidate.get("missing_skills", []),
                "trust_score": candidate.get("trust_score", 0),
                "summary": candidate.get("summary", ""),
            }
        )
        for chunk in candidate.get("evidence_chunks", []):
            evidence_pool.append(
                {
                    "candidate": candidate.get("name", candidate.get("filename", "Unknown")),
                    "section": chunk.get("section", "other"),
                    "text": chunk.get("text", ""),
                }
            )

    top_evidence = retrieve_top_k(question, evidence_pool, k=6)
    evidence_block = "\n\n".join(
        f"Candidate: {item.get('candidate')} | Section: {item.get('section')}\n{item.get('text')}"
        for item in top_evidence
    )

    prompt = f"""You are an HR assistant helping an employer understand ranking results.
Answer only using candidate summaries and retrieved evidence.
If the evidence is insufficient, say so honestly.

Job Description:
{jd_text[:1800]}

Candidate Summaries:
{candidate_summary}

Retrieved Evidence:
{evidence_block}

Employer Question:
{question}
"""

    try:
        return ollama_chat(prompt)
    except Exception as e:
        return f"Error generating chatbot response: {str(e)}"



def score_candidate(candidate_name: str, cv_text: str, jd_text: str) -> Dict[str, Any]:
    lexical_similarity = tfidf_similarity(jd_text, cv_text)
    jd_profile = extract_requirements(jd_text)
    cv_profile = extract_cv_profile(cv_text, jd_profile, jd_text)
    semantic_similarity = semantic_similarity_from_chunks(
        build_retrieval_query(jd_profile, jd_text),
        cv_profile.get("cv_chunks", []),
        top_k=3,
    )
    result = compare_and_score(
        jd_profile,
        cv_profile,
        lexical_similarity,
        semantic_similarity,
        cv_text,
        jd_text,
    )
    explanation = get_explanation(cv_profile, jd_profile, result)

    name = cv_profile.get("candidate_name", candidate_name)
    if not name or name.lower() in ["unknown", "none", ""]:
        name = candidate_name.replace(".pdf", "").replace(".docx", "").replace(".txt", "").replace("_", " ").strip()

    return {
        "name": name,
        "filename": candidate_name,
        "score": result["total"],
        "scores": {
            "skills": round((result["skills_score"] / 40) * 100),
            "experience": round((result["exp_score"] / 30) * 100),
            "education": round((result["edu_score"] / 15) * 100),
            "projects": round((result["proj_score"] / 15) * 100),
        },
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
        "lexical_similarity": result["lexical_similarity"],
        "semantic_similarity": result["semantic_similarity"],
        "similarity": result["blended_similarity"],
        "trust_score": result["trust_score"],
        "is_suspicious": result["is_suspicious"],
        "summary": (
            f"Blended similarity: {result['blended_similarity']}% | "
            f"Lexical: {result['lexical_similarity']}% | Semantic: {result['semantic_similarity']}% | "
            f"{result['cv_years']} yrs exp | Relevant role: {result['role_relevant']}"
        ),
        "explanation": explanation,
        "cv_profile": cv_profile,
        "jd_profile": jd_profile,
        "evidence_chunks": cv_profile.get("retrieved_chunks", []),
        "result_raw": result,
    }
