import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

# Job dataset
jobs = {
    "Job_Title": [
        "Data Scientist",
        "Data Analyst",
        "Machine Learning Engineer",
        "Web Developer",
        "Software Engineer"
    ],
    "Description": [
        "Python machine learning statistics data analysis pandas numpy",
        "SQL Excel data visualization Power BI statistics",
        "Python deep learning NLP TensorFlow PyTorch",
        "HTML CSS JavaScript React frontend development",
        "Java C++ data structures algorithms problem solving"
    ]
}

df = pd.DataFrame(jobs)

def recommend_jobs(resume_text):
    corpus = df["Description"].tolist()
    corpus.append(resume_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    scores = list(enumerate(similarity[0]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in sorted_scores:
        results.append((df.iloc[i]["Job_Title"], round(score * 100, 2)))

    return results
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def get_missing_skills(resume, job_desc):
    resume_words = set(clean_text(resume))
    job_words = set(clean_text(job_desc))
    missing = job_words - resume_words
    return list(missing)[:5]
def skill_match_percentage(resume, job_desc):
    resume_words = set(clean_text(resume))
    job_words = set(clean_text(job_desc))
    matched = resume_words & job_words
    return int(len(matched) / len(job_words) * 100)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)  # remove special chars
    return text.split()

# UI
st.markdown("""
# 💼 AI Job Recommender  
### 🚀 Find your best career match instantly  

Upload your resume and get:
- 🎯 Top job matches  
- ⚠️ Missing skills  
- 💡 Improvement suggestions  
""")

# 🎨 Small UI styling
st.markdown("""
<style>
    .stProgress > div > div > div {
        height: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

resume_text = ""

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume uploaded successfully!")

if st.button("🔍 Find Jobs"):
    if resume_text:
        results = recommend_jobs(resume_text)

        st.markdown("## 🎯 Your Best Career Matches")

        top_job = results[0][0]

        for i, (job, score) in enumerate(results[:3]):

            rank = i + 1

            # 🔹 Section Title
            st.markdown(f"### 🔹 Job {rank}")

            # 🏆 Highlight best match
            if job == top_job:
                st.success(f"🏆 Rank #{rank}: {job} - {score}%")
            else:
                st.write(f"#{rank} 💼 {job} - {score}% match")

            # 📊 Progress bar
            st.progress(score / 100)

            job_desc = df[df["Job_Title"] == job]["Description"].values[0]

            # 🔍 Skill match %
            match_percent = skill_match_percentage(resume_text, job_desc)
            st.write(f"🔍 Skill Match: {match_percent}%")

            # ❌ Missing skills
            missing = get_missing_skills(resume_text, job_desc)
            st.error(f"❌ Missing Skills: {', '.join(missing)}")

            # 💡 Suggestion
            st.info("💡 Tip: Learn these skills to improve your chances")

            st.write("---")

        # 🎯 Final Recommendation
        st.success(f"🎯 Recommended Role: {top_job}")

    else:
        st.warning("⚠️ Please upload a resume")

# 👨‍💻 Footer
st.markdown("---")