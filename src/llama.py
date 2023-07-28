from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pytube import YouTube
import ast
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    return tokenizer, model

tokenizer, model = load_model()

embeddings_file = '/Users/jmota/Projects/hackathon-sword/notebooks/embeddings_text.csv'
df_emb = pd.read_csv(embeddings_file)
df_emb['embeddings'] = df_emb['embeddings'].apply(ast.literal_eval)
embeddings = df_emb['embeddings'].apply(np.array)


def calculate_embedding(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')

    with torch.no_grad():
        embeddings = model(input_ids)[0]

    mean_embedding = torch.mean(embeddings, dim=1).squeeze()

    return mean_embedding.numpy()


def get_most_similar_videos(title, text):
    
    new_embedding = calculate_embedding(text)
    cosine_sim = cosine_similarity(new_embedding.reshape(1, -1), list(embeddings))
    top_6_indices = np.argsort(cosine_sim.flatten())[-11:][::-1]
    top_6_urls = df_emb.loc[top_6_indices, 'url']
    
    count = 0
    for i in top_6_urls:
        yt = YouTube(i)
        if yt.title != title:
            with st.expander(label='', expanded=True):
                st.write(f"""
                    **Title**: {yt.title}

                    **Views**: {yt.views}

                    **Duration**: {np.round(yt.length / 60, 2)} min
                    """
                )
                st.image(yt.thumbnail_url, width=400, caption=i)
                count +=1
        

