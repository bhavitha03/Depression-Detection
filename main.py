import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import base64
import nltk  # Import the nltk module
from nltk.tokenize import word_tokenize  # Import word_tokenize
from nltk.corpus import stopwords

# Check if NLTK's 'stopwords' resource is available, and if not, download it
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    st.info("Downloading NLTK stopwords data. This may take a moment...")
    nltk.download('stopwords')
    st.success("NLTK stopwords data downloaded successfully.")

# Check if NLTK's 'punkt' resource is available, and if not, download it
try:
    nltk.data.find('tokenizers/punkt/PY3/english.pickle')
except LookupError:
    st.info("Downloading NLTK 'punkt' data. This may take a moment...")
    nltk.download('punkt')
    st.success("NLTK 'punkt' data downloaded successfully.")



# Load your dataset
data = pd.read_csv("tweets.csv")

# Data Preprocessing
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in set(stopwords.words('english'))]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

data['message'] = data['message'].apply(preprocess_text)

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
#st.title("Depression Prediction")
#st.write("Enter a message to predict if it indicates depression or not.")

st.markdown("""
<style>
.title {
    font-size: 70px;
    color: black;
}
input[type="text"] {
    font-size: 20px;
}
.message-label {
    font-size: 50px; /* Adjust the font size as needed */
    color: black;
}
.red-text {
    color: red; /* Set text color to red */
}
.green-text {
    color: green; /* Set text color to green */
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Depression Prediction</p>', unsafe_allow_html=True)

#user_input = st.text_input("Enter a message:")
st.markdown('<p class="message-label">Enter a message:</p>', unsafe_allow_html=True)
user_input = st.text_input("")

if st.button("Predict"):
    user_input = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input])
    prediction = clf.predict(user_input_vec)
    if prediction[0] == 1:
        st.markdown('<p class="red-text" style="font-size: 50px;">Predicted: Depressed</p>', unsafe_allow_html=True)

    else:
        st.markdown('<p class="green-text" style="font-size: 50px;">Predicted: Not Depressed</p>', unsafe_allow_html=True)

import base64
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
add_bg_from_local('impde.jpg')
