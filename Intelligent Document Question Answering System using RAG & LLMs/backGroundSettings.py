"""---------------------------------------------------------------------------------------------------------------
                Project Name : Intelligent Document Question Answering System
                               using RAG and Large Language Models
                Author       :  Vaishali Jorwekar
                Date         :  2 Jun 2026 
                File         :  backGroundSettings.py
-------------------------------------------------------------------------------------------------------------------
Problem statement   : This project allows the user to upload a PDF document and ask
# questions based on its content. The system uses RAG architecture,
# FAISS vector database, Sentence Transformers embeddings, and a
# local LLM through Ollama to generate context-aware answers.
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os
#os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import platform
import subprocess
#####################################################################################################    
#   Function Name   :   initializeSessionVariables
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Initializes Session variables across streamlit application
#   Author          :   Vaishali M. Jorwekar 
#   Date             :   4 Jun 2026             
#####################################################################################################    
def initializeSessionVariables():
    sessionVar={
        "ollama_process" : False
    }
    for key, val in sessionVar.items():
        if key not in st.session_state:
            st.session_state[key] = val
#############################################################################################
#   Function Name    :  fontAndBackgroundSettings
#   Description      :  font And Background Settings function
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   4 Jun 2026
#############################################################################################
def fontAndBackgroundSettings():
      st.markdown("""
<style>

/* Main Background */

.stApp {
    background: linear-gradient(to right, #fdf4ff, #fae8ff);
    color: #3b0764;
}

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
    text-align: center;
    text-shadow: 2px 2px 5px black;
    margin-top: 10px;
}
/* Sidebar */


/* Buttons */

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)          
#############################################################################################
#   Function Name    :  pageSettings
#   Description      :  Page Setting function
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def pageSettings():
    #   Add sidebar
    with st.sidebar:
        col1,col2=st.columns(2)
        st.subheader("Ollama Start/Stop")

        with col1:
            startDisabled=bool(st.session_state.ollama_process)
            st.button("Start Llama 3",on_click=startOllama,disabled= startDisabled)
            
        with col2:  
            stopDisabled=not st.session_state.ollama_process
            st.button("Stop Llama 3",on_click=stopOllama,disabled=stopDisabled)
            
        st.divider()
        
        st.subheader("📘 Project Information")

        st.write("This project demonstrates a complete RAG-based PDF Question Answering System.")
        st.subheader(" 💻 Technologies Used")
        st.write("""
        - Python
        - Streamlit
        - PyPDF2
        - Sentence Transformers
        - FAISS
        - Ollama
        - Llama3
        """)
        st.subheader("Concepts Covered")
        st.write("""
        - PDF Processing
        - Text Chunking
        - Embeddings
        - Vector Database
        - Semantic Search
        - RAG
        - LLM
        - Prompt Engineering
        """)

        
#############################################################################################
#   Function Name    :  startOllama
#   Description      :  Start Ollma lamma3 locally
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def startOllama():        
    try:
        if not st.session_state.ollama_process:
            currentOS = platform.system()
            applescript_cmd = 'tell application "Terminal" to do script "ollama run llama3"'
            
            subprocess.Popen(["osascript", "-e", applescript_cmd])
            #proc = subprocess.Popen(["ollama", "run", "llama3"])
            st.session_state.ollama_process = True  
            st.sidebar.success("Ollama run command sent successfully.")
        else:
            st.sidebar.info("Process is already running.")
    except Exception as e:
        st.sidebar.error(f"Failed to start Ollama: {e}")

#############################################################################################
#   Function Name    :  startOllama
#   Description      :  Stop Ollama  lamma3 locally
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   2 Jun 2026
#############################################################################################
def stopOllama():        
    try:
        if st.session_state.ollama_process:
            subprocess.run(["ollama", "stop", "llama3"], check=True)
            
            subprocess.run(
                [
                   "pkill",
                    "-f",
                    "ollama run llama3",
                ],
                check=True,
            )
            #subprocess.run(
            #    [
            #        "osascript",
            #        "-e",
            #        'tell application "Terminal" to close (every window whose contents contains "ollama")',
            #    ],
            #    check=True,
            #)
            st.session_state.ollama_process = False
            st.sidebar.success("Ollama process terminated.")
            
        else:
            st.sidebar.info("Ollama not running.")
    except Exception as e:
        st.sidebar.error(f"Failed to stop Ollama: {e}")
