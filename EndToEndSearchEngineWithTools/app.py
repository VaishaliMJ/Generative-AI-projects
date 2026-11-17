"""------------------------------------------------------------------------------
        Problem Statement   :   Streamlit application with Search Engine and Tools 
        File Name           :   app.py
        Name                :   Vaishali M. Jorwekar
        Date                :   5 Nov 2025
------------------------------------------------------------------------------"""
###############################################################################
#   Import Libraries
###############################################################################
import streamlit as st

from langchain_community.tools import (
                                        ArxivQueryRun,
                                        WikipediaQueryRun,
                                        DuckDuckGoSearchRun
                                    )
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper

#from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper

from langchain_groq import ChatGroq

#from langchain.agents import initialize_agent,AgentType
##from langchain_classic.agents import create_tool_calling_agent
#from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import AgentType, initialize_agent
#from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv


###############################################################################
#   Function        :   toolsCreation
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Creates tools for the Streamlit application
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
###############################################################################

def toolsCreation():
    """Search Wikipedia,Arxiv and DuckDuckGoSearchRun for information about a given topic."""

    #   Inbuilt tools
    apiWrapperWikipedia=WikipediaAPIWrapper(top_k_results=1,doc_content_char_max=200)  
    
    wikiTool=WikipediaQueryRun(api_wrapper=apiWrapperWikipedia)
    
    apiWrapperArxiv=ArxivAPIWrapper(top_k_results=1,doc_content_char_max=200)
    
    arxivTool=ArxivQueryRun(api_wrapper=apiWrapperArxiv)
    
    searchGoogleTool=DuckDuckGoSearchRun(name="Search",description="Search the web for information")
    
    
    
    #arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    #arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    #api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
    #wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

    #search=DuckDuckGoSearchRun(name="Search")"""

    return [searchGoogleTool,wikiTool,arxivTool]
###############################################################################
#   Function        :   getLLMModel
#   Input Params    :   groq_API_key
#   Output Params   :   None
#   Description     :   Load the LLM model using Ollama
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
###############################################################################
def getLLMModel(groq_API_key):
    llm=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_API_key)
    return llm  
   
###############################################################################
#   Function        :   setStreamlitPageConfig
#   Input Params    :   groq_API_key,tools
#   Output Params   :   None
#   Description     :   Set the Streamlit page configuration
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
###############################################################################
def setStreamlitPageConfig(tools,groq_API_key):
    #response=""
    st.title(" 🔎 LangChain : Chat and Search ")
    
    #   Sidebar
    st.sidebar.title("Settings")
        
    if "messages" not in st.session_state:
        st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi,I am a chatbot who can search the web. How can I assist you today?"
        }
        ]
     
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])
    
        
    if prompt:=st.chat_input("Ask a question or type 'exit' to end the chat:"):
        st.session_state.messages.append({"role": "user","content": prompt})  
        
        
        st.chat_message("user").write(prompt)
         #   Load the LLM model using the GROQ API key
        llm=getLLMModel(groq_API_key)
        
        search_agent=initialize_agent(tools,llm,
                                      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      handle_parsing_errors=True)

        
        with st.chat_message("assistant"):
            
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)  
            #response = search_agent.run({"input": prompt},config={"callbacks": [st_cb]})
            response = ""
            try:
                response=search_agent.run(input=prompt,callbacks=[st_cb])
                #response=search_agent.invoke({"input": prompt},config={"callbacks": [st_cb]})
                if response is None:
                    response = "The agent returned an empty response."

            except Exception as e:
                st.error(f"Agent execution failed: {repr(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            st.write(response)
                                             
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 Nov 2025
###############################################################################
def main():
    #   load environment variables from .env file
    load_dotenv()
    
    groq_API_key = os.getenv("GROQ_API_KEY")
    
    #   Tools Creation
    tools=toolsCreation()
    
    #   Streamlit setup
    setStreamlitPageConfig(tools,groq_API_key)
    
    

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()