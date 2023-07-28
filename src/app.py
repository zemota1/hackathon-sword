import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import sys
import json
import ast
from langchain.tools import YouTubeSearchTool
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
sys.path.insert(0, ".")
sys.path.insert(1, "..")
from src.langchain_code import process_summary, search_google, search_wikipedia, search_arxiv
from src.llama import get_most_similar_videos



st.write("""
    ## YouTube Search Assistant
    """)


#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
@st.cache_resource
def fetch_youtube_videos(search_text, number):
    tool = YouTubeSearchTool()
    all_videos = tool.run(f"{search_text},{number}")
    all_videos = ast.literal_eval(all_videos)
    all_titles = list()
    all_titles_transcription = dict()
    all_trans = dict()

    for each_video in all_videos:
        try:
            id = each_video.split("watch?v=")[1].split("&")[0]
            yt = YouTube(f"https://www.youtube.com/{each_video}")

            transcription = YouTubeTranscriptApi.get_transcript(id)
            transcription_json = json.dumps(transcription)
        except:
            all_trans[each_video] = False
            continue
        if np.round(yt.length / 60, 2) > 20:
            all_trans[each_video] = False
            continue

        all_titles.append(yt.title)
        all_titles_transcription[yt.title] = transcription_json
        all_trans[each_video] = True

    return all_videos, all_titles, all_titles_transcription, all_trans

col_1, col_2, col_3 = st.columns(3)
all_videos = list()
with col_1:
    col_1_1, col_1_2 = st.columns([0.85, 0.15])

    with col_1_1:
        text_input = st.text_input(
            "Seach topic"
        )


    if text_input != '':
        all_videos, all_titles, all_titles_transcription, all_trans = fetch_youtube_videos(text_input, 10)

        for each_video in all_videos:
            
            if all_trans[each_video]:
            
                id = each_video.split("watch?v=")[1].split("&")[0]
                yt = YouTube(f"https://www.youtube.com/{each_video}")
                with st.expander(label='', expanded=True):
                    st.write(f"""
                        **Title**: {yt.title}

                        **Views**: {yt.views}

                        **Duration**: {np.round(yt.length / 60, 2)} min
                        """
                    )
                    st.image(yt.thumbnail_url, width=450, caption=f"https://www.youtube.com/{each_video}")

with col_2:
    
    _, col_2_2 = st.columns([0.1, 0.9])
    
    with col_2_2:

        if len(all_videos) > 0:

            video_to_summary = st.selectbox(
                'Select Video',
                tuple(all_titles))

            model_response = process_summary(all_titles_transcription[video_to_summary])
            
            content = json.loads(model_response)

            with st.expander(label='Summary', expanded=True):
                st.write(f"""
                    {content['Summary']}
                """
                )
            
            if content['Wikipedia'] != 'NA':
            
                with st.expander(label='Wikipedia search', expanded=False):
                    st.write(f"""
                        {search_wikipedia(content['Wikipedia'])}
                    """
                    )
            
            if content['Arxiv'] != 'NA':
            
                with st.expander(label='Arxiv Seach', expanded=False):
                    search_arxiv(content['Arxiv'])

with col_3:
    
    _, col_3_2 = st.columns([0.1, 0.9])
    
    with col_3_2:
    
        if len(all_videos) > 0 and video_to_summary is not None:
        
            st.write("Similar Videos")
            
            st.write(get_most_similar_videos(video_to_summary, content['Summary']))
    

        
        
