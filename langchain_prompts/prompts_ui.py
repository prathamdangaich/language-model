from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="ibm-granite/granite-4.2-30b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

country = st.selectbox("Choose Country:", ["India", "China", "Japan", "South Korea","USA", "UK", "Australia"])
games = st.selectbox("Choose Game:", ["Olympics", "Asian Games", "Commonwealth Games"])
year = st.text_input("Enter year")

template = PromptTemplate(
    template="""
    Please give me deatils of medals won by {Country} in {Games} in {Year}.
    If {Country} does not play {Games} then simply give " This country does not participate in this game".
    If {Games} was not held in {Year}. Then say {Games} was not held in {Year} and give me the closest year in which games were held.
    """,
    input_variables=['Country','Games','Year']
)

# Now filling the placeholders
prompt = template.invoke(
    {
        'Country':country,
        'Games':games,
        'Year':year
    }
)

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)