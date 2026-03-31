import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Initialize NLP tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Page configuration
st.set_page_config(page_title="Faculty Feedback Analyzer", layout="wide")
st.title("📊 Faculty Feedback Analyzer")
st.write("Upload feedback documents and analyze positive/negative sentiment with keyword extraction")

# ============================================================
# LOAD SENTIMENT WORD LISTS WITH FALLBACK
# ============================================================

# Comprehensive Fallback Lists for Streamlit Cloud (if files not found)
FALLBACK_POSITIVE_WORDS = {
    'excellent', 'good', 'great', 'amazing', 'fantastic', 'wonderful', 'love', 'loved', 'loving',
    'best', 'better', 'brilliant', 'outstanding', 'perfect', 'beautiful', 'lovely', 'awesome',
    'superb', 'exceptional', 'phenomenal', 'incredible', 'terrific', 'magnificent', 'fabulous',
    'marvelous', 'divine', 'delightful', 'pleasing', 'enjoyable', 'delicious', 'charming',
    'attractive', 'impressive', 'remarkable', 'splendid', 'glorious', 'wonderful', 'superior',
    'admirable', 'commendable', 'praiseworthy', 'worthwhile', 'valuable', 'beneficial', 'helpful',
    'useful', 'productive', 'effective', 'efficient', 'successful', 'accomplished', 'achieved',
    'thriving', 'prosperous', 'flourishing', 'blossoming', 'growing', 'improving', 'advancing',
    'progress', 'victory', 'triumph', 'winner', 'success', 'achievement', 'accomplishment',
    'excellent', 'superb', 'great', 'nice', 'good', 'wonderful', 'fantastic', 'brilliant',
    'outstanding', 'beautiful', 'lovely', 'amazing', 'awesome', 'perfect', 'great', 'wonderful',
    'fine', 'superior', 'best', 'top', 'premium', 'quality', 'reliable', 'trusted', 'safe',
    'secure', 'stable', 'strong', 'powerful', 'effective', 'efficient', 'fast', 'quick',
    'easy', 'simple', 'clear', 'bright', 'happy', 'joy', 'delight', 'pleasure', 'grateful',
    'thankful', 'blessed', 'fortunate', 'lucky', 'satisfied', 'pleased', 'content', 'happy'
}

FALLBACK_NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'worse', 'worst', 'hate', 'hated', 'hating',
    'dislike', 'disliked', 'disliking', 'failure', 'failed', 'failing', 'problem', 'problems',
    'issue', 'issues', 'difficult', 'challenging', 'hard', 'tough', 'nasty', 'ugly', 'disgusting',
    'dreadful', 'appalling', 'atrocious', 'abrasive', 'abusive', 'aggressive', 'angry', 'anxious',
    'annoying', 'annoyed', 'arrogant', 'ashamed', 'afraid', 'broken', 'chaotic', 'complex',
    'confusing', 'corrupt', 'rude', 'crude', 'cruel', 'dangerous', 'dark', 'dead', 'deadly',
    'deceitful', 'defeated', 'defiant', 'deficient', 'degrading', 'demoralize', 'depress',
    'depressing', 'despair', 'desperate', 'despise', 'despised', 'destroy', 'destructive',
    'detrimental', 'devoid', 'difficult', 'disagree', 'disagreement', 'disabled', 'disadvantage',
    'disappoint', 'disappointed', 'disappointing', 'disapproval', 'discourage', 'discouraging',
    'discord', 'discriminate', 'disgrace', 'disgraceful', 'disgust', 'disgusting', 'disheartening',
    'dishonest', 'disillusioned', 'dislike', 'disliked', 'disloyal', 'dismal', 'dismissive',
    'disobedient', 'disorder', 'disorganized', 'disoriented', 'disrespect', 'disrespectful',
    'disruptive', 'dissatisfied', 'distaste', 'distasteful', 'distress', 'distressing',
    'distrustful', 'disturbing', 'divisive', 'doubt', 'doubtful', 'dreadful', 'dumb', 'dull',
    'error', 'errors', 'evil', 'fail', 'failed', 'failure', 'fault', 'faults', 'faulty', 'fear',
    'fearful', 'flawed', 'flaw', 'fool', 'foolish', 'forbidden', 'fraud', 'fraudulent', 'frustration',
    'garbage', 'gloomy', 'gloom', 'grief', 'grim', 'gross', 'harsh', 'hate', 'hateful', 'hazard',
    'horrible', 'hurt', 'hurtful', 'ignorant', 'ill', 'illness', 'immoral', 'impossible', 'inadequate',
    'inappropriate', 'incomplete', 'incorrect', 'indifferent', 'inferior', 'injury', 'insane',
    'insensitive', 'insincere', 'insufficient', 'insult', 'insulting', 'intolerable', 'invalid',
    'irrelevant', 'irritate', 'irritating', 'irresponsible', 'jealous', 'joyless', 'judgmental'
}

@st.cache_data
def load_word_list(filepath, is_positive=True):
    """Load words from txt file with comprehensive fallback mechanism"""
    try:
        # Try multiple path combinations
        possible_paths = [
            filepath,
            os.path.join(SCRIPT_DIR, filepath),
            os.path.join(os.getcwd(), filepath),
        ]
        
        full_path = None
        for path in possible_paths:
            if os.path.exists(path):
                full_path = path
                break
        
        if full_path:
            # Try different encodings to handle various text formats
            for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(full_path, 'r', encoding=encoding) as f:
                        words = [word.strip().lower() for word in f.readlines() if word.strip()]
                    if words:  # Only return if we got words
                        return set(words)
                except (UnicodeDecodeError, UnicodeError):
                    continue
    except Exception as e:
        pass  # Silently fail and use fallback
    
    # Use fallback without warning
    return FALLBACK_POSITIVE_WORDS if is_positive else FALLBACK_NEGATIVE_WORDS

positive_words = load_word_list("Words/positive-words.txt", is_positive=True)
negative_words = load_word_list("Words/negative-words.txt", is_positive=False)

st.sidebar.info(f"📚 Sentiment Dictionary Loaded:\n- Positive: {len(positive_words)} words\n- Negative: {len(negative_words)} words")

# ============================================================
# FILE UPLOAD
# ============================================================
st.header("📂 Step 1: Upload Feedback Document")

# Create sample data for quick testing
sample_data = {
    'feedback': [
        'The course was excellent and very helpful. The instructor was knowledgeable.',
        'Great content but the pace was too fast. Difficult to follow.',
        'Amazing learning experience! Highly recommended for everyone.',
        'The material was confusing and not well organized.',
        'Wonderful course with practical examples. Very satisfied!',
        'Poor organization and lack of clarity in instructions.',
        'Fantastic professor! Very engaging and interactive.',
        'Disappointing experience. Expected better quality.'
    ]
}

# Layout for upload options
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload feedback file (Excel, CSV, or TXT)",
        type=["xlsx", "xls", "csv", "txt"],
        help="Supported formats: Excel (.xlsx, .xls), CSV (.csv), Text (.txt)"
    )

with col2:
    st.write("")
    st.write("")
    if st.button("📊 Load Sample File", use_container_width=True):
        st.session_state.use_sample = True

# Check if we should use sample data
use_sample = st.session_state.get('use_sample', False)

if uploaded_file is None and not use_sample:
    st.info("👉 Please upload a feedback file or click 'Load Sample File' to begin analysis")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================
try:
    if use_sample:
        df = pd.DataFrame(sample_data)
        st.success(f"✅ Sample file loaded! ({len(df)} rows)")
    elif uploaded_file:
        if uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "text/plain":
            # For plain text, treat each line as a feedback
            text_content = uploaded_file.read().decode('utf-8')
            feedbacks = [line.strip() for line in text_content.split('\n') if line.strip()]
            df = pd.DataFrame({'feedback': feedbacks})
        else:
            st.error("Unsupported file format")
            st.stop()
        st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
    else:
        st.error("Please upload a file or load sample data")
        st.stop()
except Exception as e:
    st.error(f"Error loading file: {str(e)}")
    st.stop()

# Show preview
st.subheader("📄 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# COLUMN SELECTION
# ============================================================
st.header("Step 2: Select Feedback Column")
column = st.selectbox(
    "Which column contains the feedback text?",
    df.columns,
    help="Select the column with student/faculty feedback"
)

feedbacks = df[column].dropna().astype(str)
st.info(f"📍 Processing {len(feedbacks)} feedbacks from '{column}' column")

# ============================================================
# NLP PREPROCESSING FUNCTION
# ============================================================
def preprocess(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [ps.stem(w) for w in tokens]
    return tokens

@st.cache_data
def analyze_feedbacks(feedbacks_list):
    """Analyze all feedbacks"""
    preprocessed = []
    all_words = []
    positive_feedback_list = []
    negative_feedback_list = []
    mixed_feedback_list = []
    
    for idx, text in enumerate(feedbacks_list):
        tokens = preprocess(text)
        preprocessed.append(" ".join(tokens))
        all_words.extend(tokens)
        
        # Sentiment classification
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        
        feedback_data = {
            'original': text,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'sentiment': 'Positive' if pos_count > neg_count else ('Negative' if neg_count > pos_count else 'Neutral')
        }
        
        if pos_count > neg_count:
            positive_feedback_list.append(feedback_data)
        elif neg_count > pos_count:
            negative_feedback_list.append(feedback_data)
        else:
            mixed_feedback_list.append(feedback_data)
    
    return preprocessed, all_words, positive_feedback_list, negative_feedback_list, mixed_feedback_list

# Run analysis
with st.spinner("🔄 Analyzing feedbacks..."):
    cleaned_texts, all_words, pos_feedbacks, neg_feedbacks, neutral_feedbacks = analyze_feedbacks(feedbacks)

# ============================================================
# RESULTS HEADER
# ============================================================
st.header("Step 3: Analysis Results")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📍 Total Feedbacks", len(feedbacks))
with col2:
    st.metric("😊 Positive", len(pos_feedbacks), f"{len(pos_feedbacks)/len(feedbacks)*100:.1f}%")
with col3:
    st.metric("😔 Negative", len(neg_feedbacks), f"{len(neg_feedbacks)/len(feedbacks)*100:.1f}%")
with col4:
    st.metric("😐 Neutral", len(neutral_feedbacks), f"{len(neutral_feedbacks)/len(feedbacks)*100:.1f}%")

# ============================================================
# KEYWORD EXTRACTION
# ============================================================
st.subheader("🔑 Keyword Analysis")

# Word frequency
word_freq = Counter(all_words)
top_n = st.slider("Number of top keywords to display:", 5, 30, 15)

# Create two columns for word frequency visualizations
col1, col2 = st.columns(2)

with col1:
    st.write("**Top Overall Keywords**")
    top_words = dict(word_freq.most_common(top_n))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    words_list = list(top_words.keys())
    counts = list(top_words.values())
    
    bars = ax.barh(words_list[::-1], counts[::-1], color='steelblue')
    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Keywords Overall', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.write("**Sentiment Distribution**")
    sentiment_counts = {
        'Positive': len(pos_feedbacks),
        'Negative': len(neg_feedbacks),
        'Neutral': len(neutral_feedbacks)
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    wedges, texts, autotexts = ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), 
                                        autopct='%1.1f%%', colors=colors, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Feedback Sentiment Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# POSITIVE KEYWORDS
# ============================================================
st.subheader("😊 Positive Feedback & Keywords")

positive_keywords = Counter()
for feedback_item in pos_feedbacks:
    tokens = preprocess(feedback_item['original'])
    for word in tokens:
        if word in positive_words:
            positive_keywords[word] += 1

if positive_keywords:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Count: {len(pos_feedbacks)} Positive Feedbacks**")
        top_pos_keywords = dict(positive_keywords.most_common(top_n))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        keywords_list = list(top_pos_keywords.keys())[::-1]
        keyword_counts = list(top_pos_keywords.values())[::-1]
        
        bars = ax.barh(keywords_list, keyword_counts, color='#2ecc71')
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Positive Keywords', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**Keyword Count Table**")
        pos_kw_df = pd.DataFrame(list(top_pos_keywords.items()), columns=['Keyword', 'Count'])
        pos_kw_df = pos_kw_df.sort_values('Count', ascending=False)
        st.dataframe(pos_kw_df, use_container_width=True, hide_index=True)
    
    # Show positive feedbacks
    with st.expander(f"📖 View All Positive Feedbacks ({len(pos_feedbacks)})"):
        for idx, feedback_item in enumerate(pos_feedbacks[:50], 1):
            st.write(f"**{idx}.**  {feedback_item['original']}")
else:
    st.info("No positive keywords found in feedbacks")

# ============================================================
# NEGATIVE KEYWORDS
# ============================================================
st.subheader("😔 Negative Feedback & Keywords")

negative_keywords = Counter()
for feedback_item in neg_feedbacks:
    tokens = preprocess(feedback_item['original'])
    for word in tokens:
        if word in negative_words:
            negative_keywords[word] += 1

if negative_keywords:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Count: {len(neg_feedbacks)} Negative Feedbacks**")
        top_neg_keywords = dict(negative_keywords.most_common(top_n))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        keywords_list = list(top_neg_keywords.keys())[::-1]
        keyword_counts = list(top_neg_keywords.values())[::-1]
        
        bars = ax.barh(keywords_list, keyword_counts, color='#e74c3c')
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Negative Keywords', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**Keyword Count Table**")
        neg_kw_df = pd.DataFrame(list(top_neg_keywords.items()), columns=['Keyword', 'Count'])
        neg_kw_df = neg_kw_df.sort_values('Count', ascending=False)
        st.dataframe(neg_kw_df, use_container_width=True, hide_index=True)
    
    # Show negative feedbacks
    with st.expander(f"📖 View All Negative Feedbacks ({len(neg_feedbacks)})"):
        for idx, feedback_item in enumerate(neg_feedbacks[:50], 1):
            st.write(f"**{idx}.**  {feedback_item['original']}")
else:
    st.info("No negative keywords found in feedbacks")

# ============================================================
# TF-IDF ANALYSIS
# ============================================================
st.subheader("⭐ TF-IDF Keyword Importance")

# Filter out empty documents for TF-IDF
non_empty_texts = [text for text in cleaned_texts if text.strip()]

if len(non_empty_texts) > 1:
    try:
        vectorizer = TfidfVectorizer(max_features=50, min_df=1, max_df=0.9, stop_words='english')
        X = vectorizer.fit_transform(non_empty_texts)
        
        features = vectorizer.get_feature_names_out()
        scores = X.toarray().sum(axis=0)
        
        tfidf_dict = dict(zip(features, scores))
        top_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if top_tfidf:
            fig, ax = plt.subplots(figsize=(10, 6))
            keywords = [item[0] for item in top_tfidf][::-1]
            tfidf_scores = [item[1] for item in top_tfidf][::-1]
            
            bars = ax.barh(keywords, tfidf_scores, color='#3498db')
            ax.set_xlabel('TF-IDF Score', fontsize=11, fontweight='bold')
            ax.set_title(f'Top {top_n} Keywords by TF-IDF Score', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                        ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No significant TF-IDF keywords found")
    except Exception as e:
        st.warning(f"⚠️ Could not perform TF-IDF analysis: {str(e)}")
else:
    st.warning("Need at least 2 feedbacks with content for TF-IDF analysis")

# ============================================================
# EXPORT RESULTS
# ============================================================
st.subheader("💾 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Positive Keywords", use_container_width=True):
        if positive_keywords:
            export_df = pd.DataFrame(list(positive_keywords.items()), columns=['Keyword', 'Count']).sort_values('Count', ascending=False)
            csv = export_df.to_csv(index=False)
            st.download_button("Download Positive Keywords CSV", csv, "positive_keywords.csv", "text/csv")

with col2:
    if st.button("📥 Export Negative Keywords", use_container_width=True):
        if negative_keywords:
            export_df = pd.DataFrame(list(negative_keywords.items()), columns=['Keyword', 'Count']).sort_values('Count', ascending=False)
            csv = export_df.to_csv(index=False)
            st.download_button("Download Negative Keywords CSV", csv, "negative_keywords.csv", "text/csv")

with col3:
    if st.button("📥 Export Full Analysis", use_container_width=True):
        summary_df = pd.DataFrame({
            'Feedback': [f['original'] for f in pos_feedbacks + neg_feedbacks + neutral_feedbacks],
            'Sentiment': [f['sentiment'] for f in pos_feedbacks + neg_feedbacks + neutral_feedbacks],
            'Positive_Count': [f['positive_count'] for f in pos_feedbacks + neg_feedbacks + neutral_feedbacks],
            'Negative_Count': [f['negative_count'] for f in pos_feedbacks + neg_feedbacks + neutral_feedbacks]
        })
        csv = summary_df.to_csv(index=False)
        st.download_button("Download Full Analysis CSV", csv, "feedback_analysis.csv", "text/csv")

st.success("✅ Analysis complete! Review the results above and export as needed.")