import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

st.set_page_config(page_title="Email Spam Classifier", layout="wide")

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_words = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return stemmed_words, filtered_words

# ✅ FIXED: Load vectorizer and model with error details
try:
    if not os.path.exists("vectorizer.pkl") or not os.path.exists("model.pkl"):
        raise FileNotFoundError("One or both pickle files are missing.")
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()

# Load CSS file
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

load_css('styles.css')

# === [Your existing code continues as-is below...] ===


# Navigation bar
# st.markdown(
#     """
#     <div class="navbar">
#         <a href="#home">Home</a>
#         <a href="#classify">Classify Spam</a>
#         <a href="#next-steps">Next Steps</a>
#         <a href="#about">About</a>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.markdown('<div class="content">', unsafe_allow_html=True)
st.markdown('<div class="section" id="classify">', unsafe_allow_html=True)
st.markdown('<h2>📝 Enter Your Message</h2>', unsafe_allow_html=True)

# Input area for message
input_sms = st.text_area("Enter the message", key="input_text", height=200)
result = None

# Define a dictionary of spam words and their alternatives
spam_alternatives = {
    "free": "complimentary",
    "win": "achieve",
    "winner": "top performer",
    "prize": "reward",
    "credit": "points",
    "offer": "opportunity",
    "limited": "exclusive",
    "urgent": "important",
    "deal": "proposal",
    "money": "funds",
    "cash": "savings",
    "bonus": "benefit",
    "earn": "gain",
    "income": "revenue",
    "investment": "contribution",
    "cheap": "affordable",
    "discount": "reduction",
    "trial": "evaluation",
    "gift": "token",
    "guarantee": "assurance",
    "promise": "commitment",
    "unlimited": "boundless",
    "extra": "additional",
    "best price": "competitive price",
    "double": "enhanced",
    "luxury": "premium",
    "bargain": "deal",
    "100%": "completely",
    "satisfaction": "fulfillment",
    "risk-free": "secure",
    "now": "currently",
    "special": "unique",
    "instant": "immediate",
    "hurry": "act soon",
    "order": "request",
    "save": "preserve",
    "buy": "purchase",
    "only": "solely",
    "cancel": "withdraw",
    "cheap": "cost-effective",
    "congratulations": "well done",
    "winner": "achiever",
    "call now": "contact us",
    "apply now": "submit your application",
    "exclusive": "select",
    "act now": "take action",
    "urgent": "pressing",
    "refinance": "adjust",
    "no obligation": "optional",
    "access": "entry",
    "100 percent": "entirely",
    "free access": "complimentary entry",
    "loan": "advance",
    "quote": "estimate",
    "cheap": "low-cost",
    "refinance": "readjust",
    "best offer": "preferred opportunity",
    "pre-approved": "pre-qualified",
    "no fees": "free of charge",
    "deal": "arrangement",
    "double your": "enhance your",
    "increase": "boost",
    "profits": "gains",
    "lottery" : "gift",
    "affordable": "economical",
    "winning": "successful",
    "urgent": "time-sensitive",
    "limited time": "short period",
    "lowest price": "most competitive price",
    "guaranteed": "assured",
    "clearance": "final sale",
    "trial": "assessment",
    "no credit check": "no qualification required",
    "debt": "liability",
    "gift card": "voucher",
    "as seen on": "featured in",
    "click here": "visit",
    "act fast": "respond quickly",
    "gift": "token",
    "subscribe": "enroll",
    "apply": "register",
    "subscribe": "opt in",
    "buy direct": "purchase directly",
    "100% free": "completely complimentary",
    "secure": "protected",
    "don’t miss": "consider",
    "unsecured credit": "open credit",
    "fast cash": "quick funds",
    "easy money": "effortless income",
    "free gift": "complimentary item",
    "amazing": "impressive",
    "low price": "economical price",
    "work from home": "remote opportunity",
    "your chance": "your opportunity",
    "meet singles": "find connections",
    "act fast": "respond promptly",
    "extra income": "supplementary income",
    "limited availability": "restricted access",
    "don’t wait": "take advantage",
    "no catch": "without limitations"
}


# Predict button
if st.button('🔍 Predict', key='predict_button') or (input_sms and st.session_state.input_text):
    with st.spinner('Processing...'):
        # Transform the input message
        transformed_sms, original_words = transform_text(input_sms)
        if not transformed_sms:
            st.warning("⚠ The input message contains no valid words after processing.")
        else:
            vector_input = tfidf.transform([" ".join(transformed_sms)])
            result = model.predict(vector_input)[0]

    # Display the highlighted message below the original input if classified as spam
    if result == 1:
        spam_words = set(transformed_sms)

        def highlight_spam_words(original_text, spam_words):
            highlighted_text = []
            for word in original_text.split():
                stemmed_word = ps.stem(re.sub(r'\W+', '', word.lower()))
                if stemmed_word in spam_words:
                    highlighted_text.append(f'<span class="highlight">{word}</span>')  # HTML span with highlight class
                else:
                    highlighted_text.append(word)
            return ' '.join(highlighted_text)

        # Generate the highlighted version of the message
        highlighted_sms = highlight_spam_words(input_sms, spam_words)

        # Display the highlighted message
        st.markdown('<h4>🔍 Message with Highlighted Spam Words</h4>', unsafe_allow_html=True)
        st.markdown(f'<div class="highlighted-text">{highlighted_sms}</div>', unsafe_allow_html=True)

        # Generate a modified version of the message by replacing spam words with alternatives
        def replace_spam_words(text, spam_words):
            modified_text = []
            for word in text.split():
                stemmed_word = ps.stem(re.sub(r'\W+', '', word.lower()))
                if stemmed_word in spam_words:
                    # Replace with an alternative if available in spam_alternatives
                    modified_word = spam_alternatives.get(word.lower(), word)
                    modified_text.append(modified_word)
                else:
                    modified_text.append(word)
            return ' '.join(modified_text)

        # Display a button to replace spam words
        if st.button("Replace Spam Words with Alternatives"):
            modified_message = replace_spam_words(input_sms, spam_words)
            st.markdown('<h4>🔄 Message with Spam Words Replaced</h4>', unsafe_allow_html=True)
            st.markdown(f'<div class="highlighted-text">{modified_message}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<h4>📝 Your Message (Not Spam)</h4>', unsafe_allow_html=True)
        st.markdown(f'<div>{input_sms}</div>', unsafe_allow_html=True)

# Result Display
if result is not None:
    result_text = "🚫 *Spam" if result == 1 else "✅ **Not Spam*"
    st.markdown(f'<div class="header-result">Result: {result_text}</div>', unsafe_allow_html=True)

# Visualization section for spam word frequency
if result == 1:
    if st.button('📊 Visualize Spam Words', key='visualize_button'):
        word_counts = Counter(transformed_sms)
        common_words = word_counts.most_common(10)

        if common_words:
            st.markdown('<div class="subheader">📈 Top 10 Spam Words in the Message</div>', unsafe_allow_html=True)
            
            # Create a bar plot of the top 10 spam words
            common_words_dict = dict(common_words)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(common_words_dict.keys()), y=list(common_words_dict.values()), ax=ax, palette='viridis')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Words')
            plt.ylabel('Count')
            plt.title('Most Common Spam Words in the Message')
            
            st.pyplot(fig)
        else:
            st.write("🔍 No significant spam words to visualize.")

st.markdown('</div>', unsafe_allow_html=True)