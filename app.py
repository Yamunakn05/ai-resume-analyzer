import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Resume Analyzer")

st.title("🧾 AI Resume Analyzer")

job_desc = st.text_area("📌 Enter Job Description")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


if st.button("🔍 Analyze Resume"):

    if uploaded_file and job_desc:

        resume_text = extract_text_from_pdf(uploaded_file)

        # ML Matching
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_desc, resume_text])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        score = round(similarity * 100, 2)

        st.subheader(f"📊 Match Score: {score}%")

        # Skill Detection
        skills_list = ["python", "machine learning", "sql",
                       "data analysis", "deep learning",
                       "java", "excel", "communication"]

        job_skills = [skill for skill in skills_list if skill in job_desc.lower()]
        resume_skills = [skill for skill in skills_list if skill in resume_text.lower()]

        matched = list(set(job_skills) & set(resume_skills))
        missing = list(set(job_skills) - set(resume_skills))

        st.write("✅ Matched Skills:", matched)
        st.write("❌ Missing Skills:", missing)

        # Strength
        if score > 75:
            strength = "Strong 💪"
        elif score > 50:
            strength = "Medium ⚖️"
        else:
            strength = "Weak ⚠️"

        st.write(f"📈 Resume Strength: {strength}")

        # Suggestions
        if missing:
            st.write("💡 Suggestions:")
            for skill in missing:
                st.write(f"- Add {skill} to your resume")

    else:
        st.error("Please upload resume and enter job description")
