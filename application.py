import pandas as pd
import streamlit as st
import openai
import os
import jsonlines

import pickle
from rank_bm25 import BM25Okapi

openai.organization = "org-eiJyreiRZUtpiu8pm6LIIA8B"
openai.api_key_path = "/roupenminassian/gpt3-codenames/blob/main/key.py"

"""
# Data Science Institute x Disability Research Network: A UTS HASS-DSI Research Project

The project involves preprocessing textual data from the Royal Commission into "Aged Care Quality and Safety", and "Violence, Abuse, Neglect and Exploitation of People with Disability" and utilising natural language processing (NLP) techniques to improve document search functionality. Initial attempts were made to create a document-fetching algorithm designed to minimise the amount of time a user may spend searching relevant information.

Please upload a file in the correct data format below; otherwise you may use an existing, preprocessed file by selecting the below box.

"""
#Load documents
input = st.file_uploader('')
    
if input is None:
    st.write("Or use sample dataset to try the application")
    sample = st.checkbox("Download sample data from GitHub")

    try:
        if sample:
            st.markdown("""[download_link](https://gist.github.com/roupenminassian/0a17d0bf8a6410dbb1b9d3f42462c063)""")
    
    except:
        pass

else:
    with open("test.txt","rb") as fp:# Unpickling
        contents = pickle.load(fp)
  
    #Preparing model
    tokenized_corpus = [doc.split(" ") for doc in contents]

    bm25 = BM25Okapi(tokenized_corpus)
    
    user_input = st.text_input('')

    if user_input is None:
        st.write('Please enter a query above.')
    
    else:
        tokenized_query = user_input.split(" ")

        doc_scores = bm25.get_scores(tokenized_query)

        if st.button('Generate Text'):
            generated_text = bm25.get_top_n(tokenized_query, contents, n=1)
            st.write(generated_text[0])
            
            GPT_text = openai.Answer.create(
  search_model="davinci",
  model="davinci",
  question=user_input,
  documents=["test.jsonl"],
  #file = "file-nYWFf5V4zKtZMv82WyakRZme",
  examples_context="In 2017, U.S. life expectancy was 78.6 years.",
  examples=[["What is human life expectancy in the United States?","78 years."]],
  max_tokens=50,
  temperature = 0.3,
  stop=["\n", "<|endoftext|>"],
)
            st.write('GPT-3 Answer:' + GPT_text['answers'][0])
