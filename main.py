import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import base64

# Check if NLTK's 'stopwords' resource is available, and if not, download it
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    st.info("Downloading NLTK stopwords data. This may take a moment...")
    nltk.download('stopwords')
    st.success("NLTK stopwords data downloaded successfully.")


# Load your dataset
data = pd.read_csv("tweets.csv")

# Data Preprocessing
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in set(nltk.corpus.stopwords.words('english'))]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Check if the 'label' column exists in your dataset
if 'label' in data.columns:
    data['message'] = data['message'].apply(preprocess_text)
else:
    st.error("The 'label' column is missing in your dataset.")

# Split the data into features (X) and labels (y)
X = data['message']
y = data['label']

# Vectorize the text data using Count Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_vec, y)

# Streamlit App
st.title("Depression Prediction")
st.write("Enter a message to predict if it indicates depression or not.")

user_input = st.text_input("Enter a message:")
if st.button("Predict"):
    if not user_input:
        st.warning("Please enter a message.")
    else:
        user_input = preprocess_text(user_input)
        user_input_vec = vectorizer.transform([user_input])
        prediction = clf.predict(user_input_vec)
        if prediction[0] == 1:
            st.write("Predicted: Depressed")
        else:
            st.write("Predicted: Not Depressed")

# Add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('imp1.jpeg')
