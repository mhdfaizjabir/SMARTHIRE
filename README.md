# SmartHire AI

A retrieval-enhanced local AI system for CV screening, candidate ranking, and recruiter decision support using Streamlit, Ollama, TF-IDF, and sentence embeddings.

---

## Overview

SmartHire AI is a local AI-powered recruitment assistant that helps recruiters and employers screen CVs against a job description, rank candidates, retrieve the most relevant evidence from each CV, generate explanations, suggest interview questions, and answer recruiter queries.

The system combines:

- Lexical matching with TF-IDF
- Semantic retrieval with sentence embeddings
- Chunk-based CV retrieval
- Rule-based candidate scoring
- LLM-based structured extraction and explanation
- Evidence-grounded recruiter Q&A

Everything runs locally using Ollama, so CVs and job descriptions do not need to be sent to external APIs.

---

## Features

- Upload multiple CVs in PDF, DOCX, or TXT
- Paste a job description
- Automatically rank candidates
- Use retrieval-enhanced grounding to focus on the most relevant CV chunks
- Compare lexical TF-IDF similarity and semantic retrieval similarity
- Extract structured candidate information:
  - skills
  - role history
  - years of experience
  - education
  - projects
- Generate:
  - candidate strengths
  - weaknesses
  - verdicts
  - tailored interview questions
- Detect suspicious skill claims using trust scoring
- Ask recruiter questions through an Employer Chat Assistant
- Export results as CSV



<img width="1778" height="697" alt="image" src="https://github.com/user-attachments/assets/7c2d2c69-e803-4797-8f18-ff0faf8ff7fe" />

<img width="1774" height="774" alt="image" src="https://github.com/user-attachments/assets/0422e8c2-9422-4b24-90fe-694c1de54c61" />

<img width="941" height="704" alt="image" src="https://github.com/user-attachments/assets/818e70da-9cec-4063-8817-eae4c61ee211" />

<img width="1416" height="492" alt="image" src="https://github.com/user-attachments/assets/14353eb8-b665-4b11-8368-355ee4fddd2f" />

---

## Tech Stack

- Frontend / UI: Streamlit
- Local LLM: Ollama (`gemma3:1b`)
- Lexical Similarity: scikit-learn TF-IDF
- Semantic Retrieval: sentence-transformers (`all-MiniLM-L6-v2`)
- Vector Math: NumPy
- Document Parsing: PyMuPDF, python-docx
- Data Handling: pandas

---

## Project Structure

```bash
.
├── app.py
├── scorer.py
├── retrieval.py
├── extractor.py
├── requirements.txt
└── README.md
