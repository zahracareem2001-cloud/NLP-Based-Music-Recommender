# recommend.py
import joblib
import logging

# Simple logging (cloud safe)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

logging.info("Loading data...")
df = joblib.load("df_cleaned.pkl")
cosine_sim = joblib.load("cosine_sim.pkl")
logging.info("Data loaded successfully.")


def recommend_songs(song_name, top_n=5):
    idx = df[df["song"].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return None

    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    song_indices = [i[0] for i in sim_scores]

    result_df = df[["artist", "song"]].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    return result_df
