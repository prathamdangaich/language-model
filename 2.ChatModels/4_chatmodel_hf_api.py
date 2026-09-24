from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="ibm-granite/granite-4.2-30b",
    task="text-generation",
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")


print(result.content)
