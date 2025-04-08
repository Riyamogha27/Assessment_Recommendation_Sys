import streamlit as st 
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("/home/riya23235/SHL/Preprocessed_shl_assessments.csv")
embeddings = torch.tensor(np.load("/home/riya23235/SHL/assessment_embeddings.npy")).to(device)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

st.title("🔍 SHL Assessment Recommendation System")

query = st.text_input("Enter job description or natural language query:")

if st.button("Get Recommendations") and query:
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)

    # Compute similarity
    scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()

    # Attach scores and sort
    df["Score"] = scores
    top_results = df.sort_values(by="Score", ascending=False).head(10)

    # Display
    for _, row in top_results.iterrows():
        st.markdown(f"### [{row['Title']}]({row['Link']})")
        st.markdown(f"**Remote:** {row['Remote Testing']} | **Adaptive:** {row['Adaptive/IRT']}")
        st.markdown(f"**Duration:** {row['Duration (In Minutes)']} mins | **Type:** {row['Test Type']}")
        st.markdown("---")
