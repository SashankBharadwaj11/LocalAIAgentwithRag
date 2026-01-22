from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv") 

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_db"
add_documents = os.path.exists(db_location)

if not os.path.exists(db_location):
    documents = []
    ids = []

    for i,row in df.iterrows():
        document = Document(
            page_content = row['Title'] + " " +row["Review"],
            metadata = {"ratings": row["Rating"], "date": row["Date"]}
        )

        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if not add_documents:
    vector_store.add_documents(documents, ids=ids)
    vector_store.persist()

# This allows you to look up k similar documents based on embeddings. The we can pass those documents to the LLM.
#This used a similarity search to find the top k most relevant documents based on the embeddings.
# And then passs those documents to the LLM for context through the langchain retriever interface.
retriver = vector_store.as_retriever(search_kwargs={"k":5})