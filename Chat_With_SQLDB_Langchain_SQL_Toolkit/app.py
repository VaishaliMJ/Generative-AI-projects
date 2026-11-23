"""-------------------------------------------------------------------------------------
Problem Statement   :   Create a SQL based Chatbot application
                
Author              :   Vaishali M. Jorwekar  

Date                :   19 Nov 2026
                
-------------------------------------------------------------------------------------"""
import streamlit as st

from pathlib import Path

#from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import create_sql_agent
#from langchain.agents import AgentType
from langchain_classic.agents import AgentType, initialize_agent
#from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
#from  langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import SQLDatabaseToolkit
#from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from sqlalchemy import create_engine

import sqlite3
from langchain_groq import ChatGroq
###############################################################################
LOCAL_DB="USE_LOCAL_DB"
MY_SQL_DB="USE_MY_SQL_DB"

###############################################################################
###############################################################################
#   Function        :   configureDatabase
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Configures the database connection based on user selection
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2016
###############################################################################
@st.cache_resource(ttl="2h")
def configureDatabase(dbURI,mySQLHost=None,mySQLUser=None,mySQLPassword=None,mySQLDatabase=None):
    if dbURI==MY_SQL_DB:
        if not mySQLHost or not mySQLUser or not mySQLPassword or not mySQLDatabase:
            st.error("Enter MySQL database connection info.")
            st.stop()
        connectionString=f"mysql+mysqlconnector://{mySQLUser}:{mySQLPassword}@{mySQLHost}/{mySQLDatabase}"
        engine=create_engine(connectionString)
        db=SQLDatabase(engine)
    else:
        dbPath=Path(__file__).parent / "Student.db"
        creator=lambda:sqlite3.connect(f"file:{dbPath}?mode=ro", uri=True)
        
        engine=create_engine(f"sqlite:///", creator=creator)
        db=SQLDatabase(engine)
    return db
    
    

###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
#   Date            :   18 Nov 2016
###############################################################################
def main():
    st.title("Chatbot with SQL Database")  

    radioOption=["Connect to local SQlite Database","Connect to MySQL Database"]
    selectedOption=st.sidebar.radio("Select Database Option",radioOption)
    
    if radioOption.index(selectedOption)==1:
        dbURI=MY_SQL_DB
        mySQLHost=st.sidebar.text_input("Enter MySQL Host")
        mySQLUser=st.sidebar.text_input("Enter MySQL User")
        mySQLPassword=st.sidebar.text_input("Enter MySQL Password",type="password")
        mySQLDatabase=st.sidebar.text_input("Enter MySQL Database Name")
    else:
        dbURI=LOCAL_DB
    groqAPIKey=st.sidebar.text_input("Enter Groq API Key",type="password")
    
    if not dbURI:
        st.sidebar.warning("Enter database info and URI.")
        return   
    if not groqAPIKey:
        st.sidebar.warning("Enter Groq API Key.")
        return  
    
    llmModel=ChatGroq(api_key=groqAPIKey, model="openai/gpt-oss-120b", 
                        temperature=0.5,streaming=True)
    
    response = llmModel.invoke("Hello, how are you?")

    print("LLM Response:", response)
    print("Content:", response.content)
    if dbURI==MY_SQL_DB:
        db=configureDatabase(dbURI,mySQLHost,mySQLUser,mySQLPassword,mySQLDatabase)
    else:
        db=configureDatabase(dbURI)
    
    #   Create SQLDatabaseToolkit with the database connection
    toolkit=SQLDatabaseToolkit(db=db, llm=llmModel)
    #   Create an agent with the SQLDatabaseToolkit
    agent = create_sql_agent(
                            llm=llmModel,
                            toolkit=toolkit,
                            verbose=True,
                            agent_type="tool-calling",
                            handle_parsing_errors=True
                        )
    if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
        st.session_state["messages"] = [{
                                "role": "assistant", 
                                "content": "Hello,How can I help you?"}]

    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])  
     
    userQuery=st.chat_input(placeholder="Ask your questions related to the database")
    if userQuery:
         st.session_state.messages.append({"role":"user","content":userQuery})
         st.chat_message("user").write(userQuery)
         
         with st.chat_message("assistant"):
             streamlitCallback=StreamlitCallbackHandler(st.container())
             response=agent.run(input=userQuery, callbacks=[streamlitCallback])
             st.session_state.messages.append({"role":"assistant","content":response})  
             st.write(response)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()
    