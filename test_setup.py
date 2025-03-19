import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

response = llm.invoke("Hello, Are you working on a project?")
print(response.content) # "Yes, I am working on a project."