import streamlit as st
import joblib
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already done
nltk.download('stopwords')

# Load stopwords
stop = stopwords.words('english')

# Load the models
tfidf_vectorizer = joblib.load('tfidfvectorizer.joblib')
model = joblib.load('passmodel.joblib')

# Load the training data to get the unique conditions
df_train = pd.read_csv('./data/drugsComTrain_raw.tsv', sep='\t')
df_train = df_train[(df_train['condition']=='Birth Control') | (df_train['condition']=='Depression') | (df_train['condition']=='High Blood Pressure')|(df_train['condition']=='Diabetes, Type 2')]

# Function to clean and preprocess the text
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    return ' '.join(meaningful_words)

# Function for extracting top drugs
def top_drugs_extractor(condition):
    df_top = df_train[(df_train['rating']>=9)&(df_train['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst

# Streamlit web application
st.set_page_config(
    page_title="Patient's Condition Classification and Drug Recommendation",
    page_icon="🩺",
    layout="centered"
)

st.title("Enter Medical Issue to Get Drug Recommendation")

# User input
user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    if user_input:
        # Preprocess the input
        clean_input = review_to_words(user_input)
        
        # Transform the input
        tfidf_input = tfidf_vectorizer.transform([clean_input])
        
        # Predict the condition
        prediction = model.predict(tfidf_input)[0]
        
        # Get the top 3 drug recommendations
        top_drugs = top_drugs_extractor(prediction)
        
        st.write(f"Condition: {prediction}")
        st.write("Top 3 Suggested Drugs:")
        for i, drug in enumerate(top_drugs, 1):
            st.write(f"{i}. {drug}")
    else:
        st.write("Please enter a review to predict the condition and get drug recommendations.")
