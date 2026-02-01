# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Logging (file optional, OK locally)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

logging.info("🚀 Starting preprocessing...")

nltk.download("punkt")
nltk.download("stopwords")

# Load and sample dataset
df = pd.read_csv("spotify_millsongdata.csv").sample(10000).reset_index(drop=True)
logging.info("✅ Dataset loaded: %d rows", len(df))

# Drop unnecessary column
df = df.drop(columns=["link"], errors="ignore")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning text...")
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Vectorization
logging.info("🔠 Vectorizing...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

# 🔥 REDUCED COSINE SIMILARITY (TOP 10 ONLY)
logging.info("📐 Calculating TOP-10 similarities...")
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

top_n = 10
similarity_dict = {}

for i in range(cosine_sim_matrix.shape[0]):
    sim_scores = list(enumerate(cosine_sim_matrix[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    similarity_dict[i] = sim_scores

# Save ONLY required files
joblib.dump(df, "df_cleaned.pkl")
joblib.dump(similarity_dict, "cosine_sim.pkl")

logging.info("💾 Reduced data saved.")
logging.info("✅ Preprocessing complete.")
