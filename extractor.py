import re
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document


SECTION_HINTS = {
    "skills": ["skills", "technical skills", "technologies", "toolkit", "competencies"],
    "experience": ["experience", "work experience", "employment", "professional experience"],
    "projects": ["projects", "academic projects", "personal projects"],
    "education": ["education", "academic background", "qualifications"],
    "certifications": ["certifications", "licenses", "courses"],
    "summary": ["summary", "profile", "objective", "about me"],
}


IMPORTANT_SECTION_ORDER = [
    "skills",
    "experience",
    "projects",
    "education",
    "certifications",
    "summary",
    "other",
]



def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def extract_text(file_path: str) -> str:
    lower_path = file_path.lower()
    if lower_path.endswith(".pdf"):
        return extract_pdf(file_path)
    if lower_path.endswith(".docx"):
        return extract_docx(file_path)
    if lower_path.endswith(".txt"):
        return extract_txt(file_path)
    return ""



def extract_pdf(file_path: str) -> str:
    chunks: List[str] = []
    with fitz.open(file_path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                chunks.append(page_text)
    return normalize_whitespace("\n".join(chunks))



def extract_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
    return normalize_whitespace("\n".join(paragraphs))



def extract_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return normalize_whitespace(f.read())



def guess_section_name(line: str) -> str:
    cleaned = normalize_whitespace(line).lower().rstrip(":")
    for section_name, hints in SECTION_HINTS.items():
        if cleaned in hints:
            return section_name
    return ""



def split_into_sections(text: str) -> Dict[str, str]:
    text = normalize_whitespace(text)
    if not text:
        return {"other": ""}

    sections: Dict[str, List[str]] = {"other": []}
    current_section = "other"

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        guessed = guess_section_name(line)
        if guessed:
            current_section = guessed
            sections.setdefault(current_section, [])
            continue

        sections.setdefault(current_section, []).append(line)

    return {
        section: normalize_whitespace("\n".join(lines))
        for section, lines in sections.items()
        if normalize_whitespace("\n".join(lines))
    }



def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
            continue

        start = 0
        while start < len(para):
            end = min(start + chunk_size, len(para))
            piece = para[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == len(para):
                break
            start += max(1, chunk_size - overlap)
        current = ""

    if current:
        chunks.append(current)

    return chunks



def build_cv_chunks(text: str, chunk_size: int = 1200, overlap: int = 180) -> List[dict]:
    sections = split_into_sections(text)
    cv_chunks: List[dict] = []

    ordered_sections = [s for s in IMPORTANT_SECTION_ORDER if s in sections]
    remaining_sections = [s for s in sections if s not in ordered_sections]

    for section_name in ordered_sections + remaining_sections:
        section_text = sections[section_name]
        for idx, chunk in enumerate(chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)):
            cv_chunks.append(
                {
                    "section": section_name,
                    "chunk_id": f"{section_name}_{idx}",
                    "text": chunk,
                }
            )

    if not cv_chunks and text.strip():
        return [{"section": "other", "chunk_id": "other_0", "text": text.strip()}]

    return cv_chunks
