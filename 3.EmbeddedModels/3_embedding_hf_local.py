from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Delhi is the capital of India",
    "Ranchi is the capital of Jharkhand",
    "London is the capital of UK"
]

vector = embedding.embed_documents(documents)

print(str(vector))