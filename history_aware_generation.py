from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

persistent_directory = 'db/chromedb'

embedding_model = GoogleGenerativeAIEmbeddings(model= "gemini-embedding-001")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space" : "cosine"})

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

chat_history = []

def ask_question(user_question):
    print(f" --- You asked: {user_question} ---")
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question")
        ] + chat_history + [HumanMessage(content=f"New question: {user_question}")]

        result= model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents")
    combined_input = f"""Based on the following documents, please answer this question {user_question}
Documents: 
{chr(10).join([f"{doc.page_content}" for doc in docs])}
Please provide the clear, helpful answer using only the information from these documents. If you can find the answer from the documents, just say I couldn't find the relevant information 
"""
    
    messages = [
    SystemMessage(content="You are a helpful assistant that provides the answers based on the documents and conversations"),
    HumanMessage(content=combined_input)
]
    result = model.invoke(messages)
    answer = result.content
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"------Answer------")
    print(f"\n {answer}")

    return answer
def start_chat():
    print("Ask me question, type quite to exit !!!")

    while True:
        question = input("\nAsk question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()

