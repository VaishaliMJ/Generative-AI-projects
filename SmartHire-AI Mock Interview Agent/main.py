"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    Main.py
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Marvellous SmartHire - AI Mock Interview Agent 
                    •  Develope an AI-powered Mock Interview System capable of conducting 
                       technical interviews and evaluating candidate answers intelligently.
                    •  Integrate Large Language Models (LLMs) to generate interview questions, 
                       analyze responses, and provide detailed feedback with scoring.
                    •  Implement AI-based answer evaluation, performance analysis, 
                       and automated report generation for interview preparation.
                    •  Apply concepts of Generative AI, Prompt Engineering, AI Agents, 
                       and conversational intelligence to simulate real interview environments.
--------------------------------------------------------------------------------------------------------"""
import streamlit as st
import pandas as pd
import time 
#####################################################################################################
from backgroundSettings import setbackground
from login import loadLoginPage
from questionBank import loadQuestionBank,setSessionQuestionBank
from studentDetails import loadStudentsDetails,setStudentDataframe
from mockTest import startMockTest
from utilityFunctions import initializeSessionVariables
#####################################################################################################

BORDER="-"*60

############################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    
    
    #if "page" not in st.session_state:
        #   Initialize session variables
    initializeSessionVariables()
    
    #   Load Students Details
    studentDF=loadStudentsDetails()
    #setStudentDataframe(studentDF)

      
    #   Load Question Bank
    questionBank=loadQuestionBank() 
    
    #setSessionQuestionBank(questionBank)
    
    #print(questionBank)  
    #st.session_state.page == "login_screen"
    #print(st.session_state.page)
    
    if (not st.session_state["logged_in"]) or (st.session_state.page == "login_screen") :
        with st.container(border=True):
            #   Set Background
            setbackground(imgWidth=500)
            #   Load Login Page
            loadLoginPage(studentDF)
            
            # Create Logger
            
            #   Question Bank
            setSessionQuestionBank(questionBank)
            #   Students Details
            setStudentDataframe(studentDF)
            
            
            time.sleep(2)
        
        
        st.stop()
    else:
        userName=st.session_state["student_Name"]
        #st.session_state.page = "start_screen"
        #setbackground(imgWidth=250)
        
        
        startMockTest(userName)  
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
