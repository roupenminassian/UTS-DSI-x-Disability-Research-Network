# Data Science Institute x Disability Research Network: A UTS HASS-DSI Research Project
## Introduction

This repository contains work conducted in collaboration with the _Data Science Institute (DSI)_ and _Disability Research Network (DRN)_ at the _University of Technology, Sydney_. 

The project involves preprocessing textual data from the Royal Commission into [_"Aged Care Quality and Safety"_](https://agedcare.royalcommission.gov.au/), and [_"Violence, Abuse, Neglect and Exploitation of People with Disability"_](https://disability.royalcommission.gov.au/) and utilising natural language processing (NLP) techniques to improve document search functionality. Initial attempts were made to create a document-fetching algorithm designed to minimise the amount of time a user may spend searching relevant information.

Our research spans various implementations of NLP techniques on this data, as well as utilising common deep-learning algorithms such as _BERT_ and [_GPT-3_](https://beta.openai.com/docs/introduction/overview). Most of our work is showcased in this repository in order for you to browse, but to also understand both the advantages and drawbacks on the applications of such algorithms in this particular use case.

We hope that with further reserarch and development, these automative tools will benefit legal professionals, as well as the general public in being able to access legal information more efficiently.

A warm thank you to [Adam Berry](https://profiles.uts.edu.au/Adam.Berry) and [Linda Steel](https://profiles.uts.edu.au/Linda.Steele) who co-supervised this topic area of research, and who have also kindly given permission to make these findings available to the public.

Feel free to also test the [current version](https://share.streamlit.io/roupenminassian/uts-dsi-x-disability-research-network/main/application.py) of our product out (created using Streamlit). It is recommended that you upload a datafile that we have processed in order for it to be successfully readable for our code. The user also has the option to adjust the temperature of the GPT-3 response (this controls how much randomness is in the output).

## Contents

1. [Data Preprocessing](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/tree/main/Data%20Preprocessing):
    - [PDF Plumber](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Data%20Preprocessing/PDF%20Plumber.py)
    - [Manual Method](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Data%20Preprocessing/Manual%20Method.py)

2. [Exploratory Data Analysis (EDA)](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Exploratory%20Data%20Analysis%20(EDA))

3. [BM25 (Retrieval Function)](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/BM25%20(Retrieval%20Function))

4. [Deep Learning Implementation](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/tree/main/Deep%20Learning%20Implementation):
    - [BERT](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Deep%20Learning%20Implementation/BERT)
    - [GPT-3](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Deep%20Learning%20Implementation/GPT-3)

5. [Importance of Data Preprocessing](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/tree/main/Importance%20of%20Data%20Preprocessing):
    - [Text Cleaning](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Importance%20of%20Data%20Preprocessing/Text%20Cleaning)
    - [Information Chunking](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Importance%20of%20Data%20Preprocessing/Information%20Chunking)
    - [Revisiting NLP Algorithms](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Importance%20of%20Data%20Preprocessing/Revisiting%20NLP%20Algorithms)
