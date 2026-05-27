import fitz
import re
import joblib
from nltk.corpus import stopwords

# load saved files
model = joblib.load('models/resume_classifier.pkl')

tfidf = joblib.load('models/tfidf_vectorizer.pkl')

label_encoder = joblib.load('models/label_encoder.pkl')

# stopwords
stop_words = set(stopwords.words('english'))


# PDF text extraction
def extract_text_from_pdf(pdf_path):

    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text


# cleaning function
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


# prediction function
def predict_resume_category(pdf_path):

    # extract text
    resume_text = extract_text_from_pdf(pdf_path)

    # clean text
    cleaned_text = clean_resume(resume_text)

    # convert to tfidf
    transformed_text = tfidf.transform([cleaned_text])

    # predict category
    prediction = model.predict(transformed_text)

    # decode label
    predicted_category = label_encoder.inverse_transform(prediction)

    return predicted_category[0]


# test prediction
sample_resume = "data/INFORMATION-TECHNOLOGY/10089434.pdf"

result = predict_resume_category(sample_resume)

print("Predicted Category:", result)
