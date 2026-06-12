## 📖 Intelligent Document Question Answering System using RAG and Large Language Models
  The Intelligent Document Question Answering System using RAG and Large Language Models is an advanced Genrative AI-Based Application that allows users to upload PDF document and ask questions based on the uploaded document content

### 🎯 Overview
The system reads the document extracts the content,converts the text into embeddings,stores embeddings into a vector database,Retrieves the most relevent information for user query and finally generates the intelligent answers using Large Language Models(LLM's)

---
## 🛠️ 💻 📚  Tech Stack
<table>
      <thead>
        <tr>
          <th>Technology</th>
          <th>Purpose</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Python</td>
          <td>Main Programming Language</td>
        </tr>
        <tr>
          <td>Streamlit</td>
          <td>Web UI</td>
        </tr>
        <tr>
          <td>PyPDF2</td>
          <td>PDF text Extraction</td>
        </tr>
        <tr>
          <td>Sentence Transformers</td>
          <td>Embedding Generation</td>
        </tr>
        <tr>
          <td>FAISS</td>
          <td>Vector Database</td>
        </tr>
        <tr>
          <td>Numpy</td>
          <td>Numerical Operations</td>
        </tr>
        <tr>
          <td>Ollma</td>
          <td>Runs Local LLM</td>
        </tr>
        <tr>
          <td>Llama3</td>
          <td>Large Language Model</td>
        </tr>
        <tr>
          <td>requests</td>
          <td>Sends API requests to Olama3</td>
        </tr>
       <tr>
          <td>RAG</td>
          <td>Retrieval and Genaration </td>
        </tr>
      </tbody>
</table>  
  
---
### 📌 Objective
To build an AI-Powered Document question answering system using Retrieval Augmented Generation(RAG) and Large Language Models(LLM)
The Project aims to
*  Upload PDF Document
*  Extract Text from the document
*  Split text into chunks
*  Generate Embeddings
*  Store Embeddings in vector database
*  Retrieve Relevent Information
*  Generate intelligent answers
*  Generate an interactive AI Based System

### 🚀 Project Workflow
*  User Uploads the PDF
*  PyPDF2 extracts the text
*  Text is divied into chunks
*  Sentence Transformers convert text into embeddings
*  FAISS vector database is used to store Embeddings
*  User enters the question as a input
*  Question is converted into embeddings
*  FAISS seraches similar vectors
*  Most relevent chunks are retieved
*  Context and question are combined into a prompt
*  Prompt is sent to Llamma3 using Ollama
*  Llama3 generates final answer
*  Answer is displayed on streamlit interface

---

#### ✍️ Author
Vaishali M. Jorwekar<br>
Date	: 12 Jun 2026
  

  
  
