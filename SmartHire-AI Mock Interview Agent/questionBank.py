"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    Question Bank Loading
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Mock Interview :    Question Bank Loading
--------------------------------------------------------------------------------------------------------"""
import streamlit as st
import pandas as pd
import mysql.connector
import logging
import random
from utilityFunctions import loadConfigFile

BORDER="-" * 65

#####################################################################################################    
#   Function Name   :   connectDatabase
#   Input Params    :   configFile
#   Output Params   :   connection
#   Description     :   Connect to MySql Database
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def connectDatabase(configFile):
    try:    
        # Establish connection
        connection = mysql.connector.connect(
            
            host=configFile["HOST"],
            user=configFile["USER"],          
            password=configFile["PASSWORD"],  
            database=configFile["DATABASE"] 
        )
    except mysql.connector.Error as error:
        print(f"Database error: {error}")
    
    return connection
#####################################################################################################    
#   Function Name   :   loadQuestions
#   Input Params    :   connection,configFile
#   Output Params   :   question List
#   Description     :   Loads questions from the list
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def loadQuestions(connection,configFile):
    try:
        if connection.is_connected():
            cursor = connection.cursor()
        
             # 1. Execute the SELECT query
            tableName=configFile["QUESTION_BANK_TABLE"]
            #query=f"SELECT id, language, complexity, question FROM {tableName} where complexity=%s"
            #cursor.execute(query, (complexityLevel,))

            query=f"SELECT id, topic, complexity, question FROM {tableName}"
            cursor.execute(query)
            
            # 2. Fetch all matching records into a Python list
            rows = cursor.fetchall()
        
            #print(f"Total rows retrieved: {len(rows)}\n")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            #print("MySQL connection closed.")   
    return rows 

#####################################################################################################    
#   Function Name   :   setSessionQuestionBank
#   Input Params    :   questionBank
#   Output Params   :   None
#   Description     :   Loads Question Bank from database
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def setSessionQuestionBank(questionBank):
    random.shuffle(questionBank)
    
    st.session_state["loaded_questions"] = questionBank
    st.session_state["current_question_index"] = 0  
    
    #st.rerun()

#####################################################################################################    
#   Function Name   :   getLogger
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This method loads logger
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def getLogger():
    if 'current_logger' in st.session_state:
        return st.session_state.get("current_logger")
    
    return logging.getLogger("fallback_console")     
#####################################################################################################    
#   Function Name   :   loadQuestionBank
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Loads Question Bank from database
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def loadQuestionBank():
    #configFile=json.load(open(CONFIG_FILE))
    configFile=loadConfigFile()
    connection=connectDatabase(configFile)
    questionBank=loadQuestions(connection,configFile)
    
    #print(f"Question Bank:{questionBank}")
    return questionBank
#####################################################################################################    
if __name__ =="__main__":
    loadQuestionBank()
#####################################################################################################    

