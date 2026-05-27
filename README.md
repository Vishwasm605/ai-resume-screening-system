->>> 🧠 VISHWAS AI Resume Intelligence

An AI-powered Resume Screening & Candidate Ranking System built using Machine Learning, NLP, Streamlit, and Analytics Dashboarding.

-> Project Overview

VISHWAS AI Resume Intelligence is an advanced ATS-style application that helps recruiters:

* Upload and analyze resumes
* Predict candidate job category
* Extract technical skills
* Match resumes with Job Descriptions (JD)
* Detect missing skills
* Generate AI recommendations
* Rank candidates automatically
* Visualize analytics dashboards

-> The system combines:

* NLP
* Machine Learning
* ATS Logic
* Resume Intelligence
* Analytics Dashboarding
* Streamlit UI

---

-> Features

Resume Classification

Predicts the candidate category using Machine Learning.

Example:
* INFORMATION-TECHNOLOGY
* HR
* FINANCE
* DESIGNER
* HEALTHCARE
* ENGINEERING

---

-> Skill Extraction

Automatically extracts important skills from resumes.

Example:

* Python
* SQL
* Machine Learning
* AWS
* React
* NLP

---

-> Job Description Matching

Calculates similarity score between:

* Resume
* Job Description

Provides:

* Match Percentage
* ATS-style Evaluation

---

-> Missing Skills Detection

Detects important missing skills for selected roles.

Example:

* TensorFlow
* Docker
* Kubernetes
* Power BI

---

-> AI Recommendations

Provides improvement suggestions for candidates.

Example:

* Add deployment projects
* Improve NLP skills
* Mention AWS experience
* Include certifications

---

-> Candidate Ranking System

Ranks multiple candidates based on:

* Resume quality
* Confidence score
* Skills
* JD match

---

-> Analytics Dashboard

Interactive analytics dashboard with:

* Total candidates
* Average score
* Top candidate
* Top skills distribution
* Candidate score comparison

---

->> Tech Stack

-> Frontend

* Streamlit
* HTML
* CSS
* Poppins Font

-> Backend

* Python

-> Machine Learning & NLP

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* NLTK

-> Data Processing

* Pandas
* NumPy

-> Visualization

* Matplotlib

-> PDF Processing

* PyMuPDF (fitz)

---

-> Project Structure

```bash
AI-Resume-Screening-System/
│
├── app/
│   └── app.py
│
├── data/
│   ├── ACCOUNTANT/
│   ├── HR/
│   ├── INFORMATION-TECHNOLOGY/
│   └── ...
│
├── models/
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   └── resume_screening.ipynb
│
├── src/
│   └── predict.py
│
├── requirements.txt
│
└── README.md
```

---

-> Machine Learning Models

Models tested:

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 53%      |
| Logistic Regression | 63%      |
| Random Forest       | 68%      |

Random Forest achieved the best performance.

---

-> Dataset

The dataset contains resumes from multiple job categories:

* ACCOUNTANT
* ADVOCATE
* AGRICULTURE
* APPAREL
* ARTS
* AUTOMOBILE
* AVIATION
* BANKING
* BPO
* BUSINESS-DEVELOPMENT
* CHEF
* CONSTRUCTION
* CONSULTANT
* DESIGNER
* DIGITAL-MEDIA
* ENGINEERING
* FINANCE
* FITNESS
* HEALTHCARE
* HR
* INFORMATION-TECHNOLOGY
* PUBLIC-RELATIONS
* SALES
* TEACHER

---

-> Installation

-> Clone Repository

```bash
git clone https://github.com/your-username/AI-Resume-Screening-System.git
```

---

-> Open Project Folder

```bash
cd AI-Resume-Screening-System
```

---

-> Create Virtual Environment

```bash
python -m venv venv
```

---

-> Activate Virtual Environment

->> Windows

```bash
venv\Scripts\activate
```

---

-> Install Requirements

```bash
pip install -r requirements.txt
```

---

-> Run Application

```bash
streamlit run app/app.py
```

Open browser:

```bash
http://localhost:8501
```

---

-> Workflow

1. Upload Resume PDF
2. Enter Candidate Name
3. Select Target Role
4. Paste Job Description
5. Click Analyze Resume
6. View:

   * Predicted Role
   * Confidence Score
   * Resume Score
   * Skills
   * JD Match
   * Missing Skills
   * AI Recommendations
   * Candidate Rankings
   * Analytics Dashboard

---

-> Future Improvements

* Semantic similarity scoring
* LLM-powered recommendations
* Resume chatbot assistant
* Cloud deployment
* Real ATS integration
* Database storage
* Multi-user recruiter system

---

-> UI Highlights

* Premium dark UI
* Responsive dashboard
* Analytics charts
* Glassmorphism cards
* Recruiter-friendly layout

---

-> Deployment Options

This project can be deployed using:

* Streamlit Cloud
* Hugging Face Spaces
* Render
* AWS EC2

---

-> Author

->> Vishwas M

AI | Machine Learning | NLP | Data Science | ATS Intelligence

---

->>> Conclusion

VISHWAS AI Resume Intelligence is a complete AI-powered ATS platform designed to automate resume screening, candidate evaluation, and recruiter analytics using Machine Learning and NLP.

This project demonstrates:

* Real-world ML application
* NLP pipeline development
* ATS workflow implementation
* Dashboard analytics
* UI/UX integration
* End-to-end AI product building
