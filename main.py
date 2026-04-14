import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample Job Dataset (you can replace with Kaggle dataset)
# -----------------------------
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

# -----------------------------
# Function to Recommend Jobs
# -----------------------------
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

# -----------------------------
# Test Example (CLI Mode)
# -----------------------------
resume = input("Enter your resume text: ")
recommendations = recommend_jobs(resume)

print("\nTop Job Matches:\n")
for job, score in recommendations:
    print(f"{job} - Match: {score}%")