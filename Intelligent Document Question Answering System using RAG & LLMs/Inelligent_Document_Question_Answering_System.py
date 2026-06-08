"""---------------------------------------------------------------------------------------------------------------
                Project Name : Intelligent Document Question Answering System
                               using RAG and Large Language Models
                Author       :  Vaishali Jorwekar
                Date         :  2 Jun 2026 
-------------------------------------------------------------------------------------------------------------------
Problem statement   : This project allows the user to upload a PDF document and ask
# questions based on its content. The system uses RAG architecture,
# FAISS vector database, Sentence Transformers embeddings, and a
# local LLM through Ollama to generate context-aware answers.
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import streamlit as st

import PyPDF2
import faiss
import numpy as np
import requests
from backGroundSettings import pageSettings,initializeSessionVariables,fontAndBackgroundSettings
from sentence_transformers import SentenceTransformer


#############################################################################################
#   Function Name    :   loadEmbeddingModel
#   Description      :   Loads Sentence Transformer model only once.
#   Input Params     :   -   
#   Output Params    :   SentenceTransformer model object
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################

@st.cache_resource
def loadEmbeddingModel():
    return SentenceTransformer("all-MiniLM-L6-v2")
#############################################################################################
#   Function Name    :  documentProcessFlow
#   Description      :  Uploads PDF and process it using LLM,shows other functionality as well
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def documentProcessFlow():
    tab1, tab2, tab3 = st.tabs([
    "Upload PDF",
    "Project Flow",
    "Technical Concepts"
    ])
    with tab1:
        uploadPDFDocument()

    with tab2:
        showProjectFlow()
    with tab3:
        showTechnicalConcepts()    
#############################################################################################
#   Function Name    :   showTechnicalConcepts
#   Description      :   Shows technical process flow of the project
#   Input Params     :   None
#   Output Params    :   None
#   Author           :   Vaishali M Jorwekar
#   Date             :   3 Jun 2026
#############################################################################################
def showTechnicalConcepts():   
    st.header("Technical Concepts Used")

    st.markdown("""
    ### 1. PDF Processing
    PyPDF2 is used to extract text from PDF pages.

    ### 2. Text Chunking
    Large PDF text is divided into smaller overlapping chunks.

    ### 3. Embeddings
    Sentence Transformers convert text chunks into numerical vectors.

    ### 4. Vector Database
    FAISS stores embeddings and performs fast similarity search.

    ### 5. Semantic Search
    User question is converted into embedding and matched with PDF chunks.

    ### 6. RAG
    Retrieval Augmented Generation retrieves relevant context before generating answer.

    ### 7. LLM
    Llama3 running locally through Ollama generates final response.

    ### 8. Prompt Engineering
    Context and question are combined into a structured prompt.
    """)
     
#############################################################################################
#   Function Name    :   extractText
#   Description      :   Reads PDF and extract text using PyPDF
#   Input Params     :   Uploaded PDF file object 
#   Output Params    :   Extracted text as string
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def extractText(uploadedPDFFile):
    text = ""

    try:
        pdfReader=PyPDF2.PdfReader(uploadedPDFFile)
        
        for page_number, page in enumerate(pdfReader.pages):
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:
        st.error(f"Error while reading PDF: {e}")

    return text

#############################################################################################
#   Function Name    :   splitTextIntoChunks
#   Description      :   Splits large PDF text into smaller overlapping chunks.
#   Input Params     :   Text, chunk size, overlap size
#   Output Params    :   List of text chunks
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def splitTextIntoChunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk=text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks 
#############################################################################################
#   Function Name    :   createVectorDatabase
#   Description      :   Converts chunks into embeddings and stores them in FAISS.
#   Input            :   Text chunks and embedding model
#   Output           :   FAISS index and embeddings
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def createVectorDatabase(chunks,embeddingModel):
    embeddings = embeddingModel.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    #print(dimension)
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings  
#############################################################################################
#   Function Name    :   searchRelevantChunks
#   Description      :   Retrieves top matching PDF chunks for user question.
#   Input            :   Question, chunks, FAISS index, embedding model, top_k
#   Output           :   List of relevant chunks
#   Author           :   Vaishali M Jorwekar
#   Date             :   3 Jun 2026
#############################################################################################
def searchRelevantChunks(question, chunks, index, embedding_model, top_k=3):
    questionEmbedding = embedding_model.encode([question])

    questionEmbedding = np.array(questionEmbedding).astype("float32")

    distances, indices = index.search(questionEmbedding, top_k)

    relevant_chunks = []

    for i in indices[0]:
        if i < len(chunks):
            relevant_chunks.append(chunks[i])

    return relevant_chunks
#############################################################################################
#   Function Name    :   extractTextFromPDF
#   Description      :   Reads PDF and extract text
#   Input Params     :   uploadedPDFFile,embeddingModel 
#   Output Params    :   -
#   Author           :   Vaishali M Jorwekar
#   Date             :   3 Jun 2026
#############################################################################################
def extractTextFromPDF(uploadedPDFFile,embeddingModel):
    if uploadedPDFFile is not None:
        st.success("PDF uploaded successfully!")
        with st.spinner("Extracting text from PDF..."):
            #   Extract Text From PDF
            pdfText = extractText(uploadedPDFFile)
            if pdfText.strip() == "":
                st.error("No text found in the uploaded PDF or Empty PDF.")
            else:
                st.subheader("Extracted Text Preview")

                st.text_area(
                    "First 2000 characters from PDF",
                    pdfText[:2000],
                    height=220
                )
            #   Split PDF data into chunks
            with st.spinner("Splitting text into chunks..."):
                chunks=splitTextIntoChunks(pdfText) 
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(f"Total Characters",len(pdfText))

            with col2:
                st.metric(f"Total Chunks",len(chunks))

            with col3:
                st.metric("Chunk Size", "500 chars") 
            with st.spinner("Creating embeddings and FAISS vector database..."):
                index, embeddings = createVectorDatabase(
                    chunks,
                    embeddingModel
                )      
            st.success("Vector database created successfully!")

            st.subheader("Ask Question from Uploaded PDF")

            question = st.text_input(
                "Enter your question"
            )    
            if question:
                with st.spinner("Searching relevant content from PDF..."):
                    relevant_chunks = searchRelevantChunks(
                        question,
                        chunks,
                        index,
                        embeddingModel,
                        top_k=3
                    )
                context = "\n\n".join(relevant_chunks)

                st.subheader("Retrieved Relevant Context")

                with st.expander("Click here to view retrieved context"):
                    st.write(context)
                with st.spinner("Generating answer using local LLM..."):
                    answer = ask_llm(question, context)
                st.subheader("Generated Answer")
                st.success(answer)    
#############################################################################################                    
#   Function Name   :   ask_llm
#   Description     :   Sends context and question to local LLM using Ollama.
#   Input Params    :   User question and retrieved context
#   Output Params   :   Answer generated by LLM   
#   Date            :   3 Jun 2026
#############################################################################################                 
def ask_llm(question, context):
    prompt = f"""
You are an intelligent document question answering assistant created for a Student

Project:
Intelligent Document Question Answering System using RAG and Large Language Models.

Instructions:
1. Answer only from the given context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I could not find the answer in the uploaded PDF."
4. Give clear and simple explanation.
5. If required, answer in bullet points.
Context:
{context}

Question:
{question}

Answer:
"""
    url = "http://localhost:11434/api/generate"
    payload= {
        "model" : "llama3",
        "prompt" : prompt,
        "stream"  : False   
    }  
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        return result["response"]

    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Please start Ollama and run: ollama run llama3"

    except Exception as e:
        return f"Error while communicating with LLM: {e}"
  
#############################################################################################
#   Function Name    :  uploadPDFDocument
#   Description      :  Uploads PDF and process it using LLM
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def uploadPDFDocument():
    st.header("Upload PDF Document")
        #   Upload PDF Document
    uploadedPDFFile = st.file_uploader(
            "Upload your PDF file here",
            type=["pdf"]
        )
    
    #   Load Embedding Model once
    embeddingModel = loadEmbeddingModel()
    
    #   Extract Text from uploaded PDF file 
    extractTextFromPDF(uploadedPDFFile,embeddingModel)
#############################################################################################
#   Function Name    :  showProjectFlow
#   Description      :  Displays RAG project flow on Streamlit UI.
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def showProjectFlow():
    st.markdown("""
    ### Complete Project Flow

    ```text
    PDF Upload
        ↓
    Extract Text from PDF
        ↓
    Split Text into Chunks
        ↓
    Convert Chunks into Embeddings
        ↓
    Store Embeddings in FAISS Vector Database
        ↓
    User Asks Question
        ↓
    Convert Question into Embedding
        ↓
    Search Similar Chunks
        ↓
    Prepare Context
        ↓
    Send Context + Question to LLM
        ↓
    Generate Final Answer
    ```
    """)


#############################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def main():
    #   Initialize Session varibales
    initializeSessionVariables()
    
    #   Page Settings
    pageSettings()
    
    #   Page Background settings
    fontAndBackgroundSettings()
    
    #   Document upload and process
    documentProcessFlow()


##############################################################################################
#   Starter
##############################################################################################
if __name__=="__main__":
    main()
    
    