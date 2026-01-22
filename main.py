from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver


model = OllamaLLM(model="llama3.2")

template = """
You are a expert in answering about anwersing questions about a pizza restaurant

Here are some relavant {reviews}

Here are somer questions about the restaurant: {questions}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model 

while True:
    print("\n\n------------------------------------\n\n")
    question = input(" Enter your question or type 'q' to quit: ")
    print("\n\n------------------------------------\n\n")
    if question.lower() == 'q':
        break

    reviews = retriver.invoke(question)
    results = chain.invoke({"reviews": reviews, "questions": question})

    print(results)