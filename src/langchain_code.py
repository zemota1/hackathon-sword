import sys
import streamlit as st
import os
import openai
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.retrievers import ArxivRetriever
sys.path.insert(0, ".")
sys.path.insert(1, "..")
from src.config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_ENGINE_NAME, SERPER_API_KEY

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = AZURE_OPENAI_KEY

model = ChatOpenAI(engine=AZURE_ENGINE_NAME, openai_api_key=AZURE_OPENAI_KEY, openai_api_base=AZURE_OPENAI_ENDPOINT, temperature=0.0)
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
retriever = ArxivRetriever(load_max_docs=5)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@st.cache_resource
def process_summary(text):
    start_message = "I'm providing you with a YouTube transcription and the title: {transcription}."
    prompt_start = """Please generate an output in JSON format. If a requested field doesn't make sense, please fill it with 'NA'. The maximum word count for each field is indicated after each field placeholder: 
                    {{
                        "Topic": "A topic derived from the video transcription, intended for use in a recommendation system (up to 10 words)",
                        "Summary": "A summary of the video transcription (up to 100 words)",
                        "Wikipedia": "Keyword to be used for deeper research on Wikipedia relating to the video (up to 5 words)",
                        "Google": "Keyword to be used for further searches on Google about the video (up to 5 words)",
                        "Arxiv": "Keyword to search for academic papers related to the video's topic (up to 5 words)"
                    }}
    
                    Transcription:
                    {text}
                """
    final_prompt = prompt_start.format(text=text)
    
    prompt = [AIMessage(content=start_message), HumanMessage(content=final_prompt)]
    model
    response = model(prompt).content

    try:
        json.loads(response)
    except ValueError as e:
        response = None

    return response

@st.cache_resource
def search_google(query):
    return search.run(query)

@st.cache_resource
def search_wikipedia(query):
    response = wikipedia.run(query)
    return response

@st.cache_resource
def search_arxiv(query):
    docs = retriever.get_relevant_documents(query=query)
    return st.json(docs[0].metadata)
