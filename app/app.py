from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import fitz
import re
import joblib
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter


st.set_page_config(
    page_title="VISHWAS AI Resume Intelligence",
    page_icon="🧠",
    layout="wide"
)


st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #07111f;
        color: white;
    }

    .main {
        background-color: #07111f;
    }

    h1, h2, h3, h4 {
        color: white;
    }

    .glass {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 25px;
        border-radius: 18px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    }

    .metric-card {
        background: #0f1f35;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #1f3557;
        text-align: center;
        color: white;
    }

    .skill-box {
        background-color: #12233d;
        color: white;
        padding: 10px 16px;
        border-radius: 12px;
        display: inline-block;
        margin: 5px;
        border: 1px solid #2d4a72;
    }

    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
    }

    .candidate-card {
        background: #0e1b2d;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #20344f;
    }

    </style>
    """,
    unsafe_allow_html=True
)


model = joblib.load('models/resume_classifier.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

stop_words = set(stopwords.words('english'))


skills_list = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "power bi",
    "tableau",
    "excel",
    "aws",
    "azure",
    "java",
    "c++",
    "react",
    "javascript",
    "html",
    "css",
    "data analysis",
    "nlp",
    "streamlit",
    "scikit-learn"
]

role_skills = {

     "ACCOUNTANT": [
        "accounting",
        "finance",
        "excel",
        "tax",
        "auditing",
        "tally",
        "bookkeeping"
    ],

    "ADVOCATE": [
        "legal",
        "law",
        "litigation",
        "contract",
        "compliance",
        "court"
    ],

    "AGRICULTURE": [
        "farming",
        "agriculture",
        "soil",
        "crop",
        "harvesting"
    ],

    "APPAREL": [
        "fashion",
        "garment",
        "textile",
        "clothing",
        "design"
    ],

    "ARTS": [
        "drawing",
        "painting",
        "creative",
        "illustration",
        "design"
    ],

    "AUTOMOBILE": [
        "automobile",
        "vehicle",
        "mechanical",
        "service",
        "repair"
    ],

    "AVIATION": [
        "aviation",
        "pilot",
        "aircraft",
        "airport",
        "flight"
    ],

    "BANKING": [
        "banking",
        "loan",
        "finance",
        "investment",
        "credit"
    ],

    "BPO": [
        "communication",
        "customer support",
        "voice process",
        "client handling"
    ],

    "BUSINESS-DEVELOPMENT": [
        "sales",
        "business development",
        "marketing",
        "client acquisition"
    ],

    "CHEF": [
        "cooking",
        "kitchen",
        "food",
        "chef",
        "restaurant"
    ],

    "CONSTRUCTION": [
        "construction",
        "civil",
        "site management",
        "autocad"
    ],

    "CONSULTANT": [
        "consulting",
        "analysis",
        "strategy",
        "client management"
    ],

    "DESIGNER": [
        "photoshop",
        "illustrator",
        "figma",
        "ui ux",
        "design"
    ],

    "DIGITAL-MEDIA": [
        "seo",
        "social media",
        "content creation",
        "marketing"
    ],

    "ENGINEERING": [
        "engineering",
        "python",
        "automation",
        "software",
        "technical"
    ],

    "FINANCE": [
        "finance",
        "investment",
        "budgeting",
        "financial analysis"
    ],

    "FITNESS": [
        "fitness",
        "trainer",
        "workout",
        "nutrition"
    ],

    "HEALTHCARE": [
        "healthcare",
        "patient care",
        "medical",
        "hospital"
    ],

    "HR": [
        "recruitment",
        "hr",
        "payroll",
        "employee relations"
    ],

    "INFORMATION-TECHNOLOGY": [
        "python",
        "sql",
        "machine learning",
        "aws",
        "cloud",
        "react",
        "java",
        "api",
        "data science",
        "nlp"
    ],

    "PUBLIC-RELATIONS": [
        "public relations",
        "branding",
        "media",
        "communication"
    ],

    "SALES": [
        "sales",
        "negotiation",
        "client handling",
        "marketing"
    ],

    "TEACHER": [
        "teaching",
        "education",
        "training",
        "classroom"
    ],

    "Data Scientist": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "sql",
        "pandas",
        "numpy",
        "scikit-learn",
        "nlp"
    ],

    "Frontend Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "typescript",
        "bootstrap"
    ],

    "Backend Developer": [
        "python",
        "django",
        "flask",
        "sql",
        "mongodb",
        "api"
    ],

    "Cloud Engineer": [
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "terraform"
    ],

    "Data Analyst": [
        "sql",
        "excel",
        "power bi",
        "tableau",
        "python"
    ]
}


def extract_text_from_pdf(uploaded_file):

    text = ""

    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    for page in pdf_document:
        text += page.get_text()

    return text


def clean_resume(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'\d{10}', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    cleaned_text = " ".join(filtered_words)

    return cleaned_text


def predict_category(resume_text):

    cleaned_text = clean_resume(resume_text)

    transformed_text = tfidf.transform([cleaned_text])

    prediction = model.predict(transformed_text)

    prediction_proba = model.predict_proba(transformed_text)

    confidence = prediction_proba.max() * 100

    predicted_category = label_encoder.inverse_transform(prediction)

    return predicted_category[0], confidence


def extract_skills(text):

    detected_skills = []

    text = text.lower()

    for skill in skills_list:

        if skill.lower() in text:
            detected_skills.append(skill)

    return detected_skills

def calculate_jd_match(resume_text, job_description):

    cleaned_resume = clean_resume(resume_text)

    cleaned_jd = clean_resume(job_description)

    vectors = tfidf.transform([
        cleaned_resume,
        cleaned_jd
    ])

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]

    return round(similarity * 100, 2)

def find_missing_skills(resume_text, selected_role):

    resume_skills = extract_skills(resume_text)

    required_skills = role_skills.get(selected_role, [])

    missing_skills = []

    for skill in required_skills:

        if skill.lower() not in [
            s.lower() for s in resume_skills
        ]:

            missing_skills.append(skill)

    return missing_skills

def generate_recommendations(
    missing_skills,
    selected_role
):

    recommendations = []

    if len(missing_skills) == 0:

        recommendations.append(
            "Excellent profile match for this role."
        )

    else:

        for skill in missing_skills:

            recommendations.append(
                f"Consider adding or improving {skill} skills."
            )

    # role-specific suggestions
    if selected_role == "Data Scientist":

        recommendations.append(
            "Include machine learning projects in resume."
        )

        recommendations.append(
            "Mention model deployment or NLP experience."
        )

    elif selected_role == "Frontend Developer":

        recommendations.append(
            "Show responsive UI/UX projects."
        )

        recommendations.append(
            "Include React portfolio links."
        )

    elif selected_role == "Backend Developer":

        recommendations.append(
            "Mention APIs and database projects."
        )

    elif selected_role == "Cloud Engineer":

        recommendations.append(
            "Add AWS/Azure certifications."
        )

    elif selected_role == "Data Analyst":

        recommendations.append(
            "Include dashboards and analytics projects."
        )

    return recommendations

selected_role = st.sidebar.selectbox(
    "Select Target Role",
    list(role_skills.keys())
)


if 'candidates' not in st.session_state:
    st.session_state.candidates = []


col1, col2 = st.columns([1,6])

with col1:
    st.markdown("# 🧠")

with col2:
    st.markdown("<h1 style='margin-top:2px;'>VISHWAS AI Resume Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:lightgray;'>Smart Resume Classification & Candidate Ranking System</p>", unsafe_allow_html=True)

st.markdown("---")


st.sidebar.title("Dashboard")

candidate_name = st.sidebar.text_input("Candidate Name")

uploaded_file = st.sidebar.file_uploader(
    "Upload Resume PDF",
    type=['pdf']
)
job_description = st.sidebar.text_area(
    "Paste Job Description",
    height=200
)


if st.sidebar.button("Analyze Resume"):

    if uploaded_file is not None:

        resume_text = extract_text_from_pdf(uploaded_file)

        prediction, confidence = predict_category(resume_text)

        skills = extract_skills(resume_text)
        jd_score = 0
        missing_skills = []
        recommendations = []

        if job_description:
            jd_score = calculate_jd_match(
                resume_text,
                job_description
            )

            missing_skills = find_missing_skills(
                resume_text,
                selected_role
            )

            recommendations = generate_recommendations(
                missing_skills,
                selected_role
            )

        score = min(100, int(confidence + len(skills) * 2))

        candidate_data = {
            'Name': candidate_name,
            'Category': prediction,
            'Confidence': round(confidence, 2),
            'Score': score,
            'Skills': ", ".join(skills)
        }


        st.session_state.candidates.append(candidate_data)

        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        st.subheader("AI Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Predicted Role</h3>
                <h2>{prediction}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Confidence</h3>
                <h2>{confidence:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Resume Score</h3>
                <h2>{score}/100</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Detected Skills")

        for skill in skills:
            st.markdown(
                f"<span class='skill-box'>✅ {skill}</span>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        
        if job_description:

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Job Description Match")

            st.progress(jd_score / 100)

            st.write(f"Match Score: {jd_score}%")

            st.subheader("Missing Skills")

            if missing_skills:

                for skill in missing_skills:

                    st.markdown(f"❌ {skill}")

            else:

                st.success(
                    "Candidate has all major required skills"
                )

            st.subheader("AI Recommendations")

            for rec in recommendations:

                st.markdown(f"💡 {rec}")


if len(st.session_state.candidates) > 0:

    st.markdown("## Top Candidate Rankings")

    candidates_df = pd.DataFrame(st.session_state.candidates)

    ranked_df = candidates_df.sort_values(
        by='Score',
        ascending=False
    )

    top_score = ranked_df.iloc[0]['Score']

    for index, row in ranked_df.iterrows():

        better_text = ""

        difference = top_score - row['Score']

        if difference == 0:
            better_text = "🏆 Best Candidate"
        else:
            better_text = f"{difference}% below top candidate"

        st.markdown(f"""
        <div class='candidate-card'>
            <h3>{row['Name']}</h3>
            <p><b>Role:</b> {row['Category']}</p>
            <p><b>Confidence:</b> {row['Confidence']}%</p>
            <p><b>Resume Score:</b> {row['Score']}/100</p>
            <p><b>Skills:</b> {row['Skills']}</p>
            <p style='color:#60a5fa;'><b>{better_text}</b></p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")

# ANALYTICS DASHBOARD

if len(st.session_state.candidates) > 0:

    st.markdown("## Analytics Dashboard")

    analytics_df = pd.DataFrame(
        st.session_state.candidates
    )

    # metrics
    total_candidates = len(analytics_df)

    average_score = analytics_df['Score'].mean()

    best_candidate = analytics_df.sort_values(
        by='Score',
        ascending=False
    ).iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Candidates",
            total_candidates
        )

    with col2:
        st.metric(
            "Average Score",
            f"{average_score:.1f}"
        )

    with col3:
        st.metric(
            "Top Candidate",
            best_candidate['Name']
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP SKILLS ANALYSIS

    all_skills = []

    for skills in analytics_df['Skills']:

        split_skills = skills.split(",")

        for skill in split_skills:

            cleaned_skill = skill.strip()

            if cleaned_skill:
                all_skills.append(cleaned_skill)

    skill_counts = Counter(all_skills)

    top_skills = dict(
        skill_counts.most_common(10)
    )
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top Skills")

        fig, ax = plt.subplots(figsize=(5,3))

        ax.bar(
            top_skills.keys(),
            top_skills.values()
        )

        plt.xticks(rotation=45, fontsize=8)

        plt.yticks(fontsize=8)

        st.pyplot(fig)

    with col2:

        st.subheader("Candidate Scores")

        fig2, ax2 = plt.subplots(figsize=(5,3))

        ax2.bar(
            analytics_df['Name'],
            analytics_df['Score']
        )

        plt.xticks(rotation=20, fontsize=8)

        plt.yticks(fontsize=8)

        st.pyplot(fig2)

st.markdown(
    "<center><p style='color:gray;'>Built by VISHWAS • AI Resume Intelligence System</p></center>",
    unsafe_allow_html=True
)
