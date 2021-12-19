#Using our manually generated list, we imported the documentation in our notebook for implementation in BM25 ranking systems.
#We asked a specific query, and based on a scoring functionality, returned the most relevant item in the list.
#You can also choose the number of most relevant documents for the function to return.

import pickle

from rank_bm25 import BM25Okapi

with open("/content/drive/MyDrive/test.txt","rb") as fp:# Unpickling
  contents = pickle.load(fp)

tokenized_corpus = [doc.split(" ") for doc in contents]

bm25 = BM25Okapi(tokenized_corpus)

query = "When have human rights been discussed in the context of COVID-19?"

tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)

bm25.get_top_n(tokenized_query, contents, n=1)
