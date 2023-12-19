import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

        input_mail = [input_sms]

        input_data_features = tfidf.transform(input_mail)

        # Making prediction
        prediction = model.predict(input_data_features)

        def calculate_risk_score(message):
            # Define rules and features to evaluate for risk score calculation
            rules = {
                'has_suspicious_words': set(["risk_word1", "risk_word2"]), 
            }

            # Initialize risk score
            risk_score = 0
            message_words = nltk.word_tokenize(message.lower())
            # Evaluate each rule and update the risk score accordingly
            for rule, value in rules.items():
                if rule == 'has_suspicious_words':
                    for word in value:
                        if word in message_words:
                            risk_score += 1  # Increase risk score if suspicious word is found
                else:
                    count = message.count(value)
                    if count > 0:
                        risk_score += count

            return risk_score

