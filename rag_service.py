import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader,  Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LLM = Ollama(model="llama3:8b")
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

def create_vector_store(document_path: str) ->  Chroma:
    print("Starting service to load docs...")
    if not os.path.exists(document_path):
        raise FileNotFoundError("Your file is not found")
    
    if document_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(document_path)
    elif document_path.lower().endswith('.txt'):
        loader = TextLoader(document_path)
    elif document_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(document_path)
    else:
        raise ValueError("The document presented has an unsupported format, Please recheck your document...")
    
    docs = loader.load()
    text_splitr = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split = text_splitr.split_documents(docs)

    store_vector = Chroma.from_documents(documents=split, embedding=EMBEDDING_MODEL)

    return store_vector

def create_rag_chain(vectorstore: Chroma):
    print("SERVICE: Creating RAG chain...")
    retriever = vectorstore.as_retriever()
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    
    return rag_chain

# This main block is for testing the service directly.
# It won't run when the module is imported by another script (like our future FastAPI app).
if __name__ == '__main__':
    print("--- Running RAG Service in Test Mode ---")
    
    # 1. Define the test document
    test_file = r"D:\Assignments\Information Systems Strategy and Management\Week 4\Week 4_Individual.docx"
    if not os.path.exists(test_file):
        print(f"Creating a dummy test file: '{test_file}'")
        with open(test_file, "w") as f:
            f.write("""The primary challenge for modern language models is not a lack of data, but rather the problem of 'hallucination'. This phenomenon occurs when a model generates text that is plausible and grammatically correct, but is factually incorrect or nonsensical. This happens because models are trained to predict the next word, not to understand truth. To combat this, techniques like Retrieval-Augmented Generation (RAG) are employed. RAG systems ground the model in external, verifiable knowledge sources before generating an answer, significantly improving factual accuracy. The system first retrieves relevant information and then uses that information as context for the language model.""")

    # 2. Create the vector store from the document
    try:
        my_vector_store = create_vector_store(test_file)
        
        # 3. Create the RAG chain
        my_rag_chain = create_rag_chain(my_vector_store)
        
        # 4. Ask a question
        question = "Where does Air India currently operate from?"
        print(f"\nTESTING: Asking question: {question}")
        
        answer = my_rag_chain.invoke(question)
        print("\nTEST RESULT:")
        print(answer)
        
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")