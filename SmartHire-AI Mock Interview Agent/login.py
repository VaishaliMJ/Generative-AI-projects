"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    Login page
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Mock Interview :    Login page
--------------------------------------------------------------------------------------------------------"""
import streamlit as st
import pandas as pd


from studentDetails import registerNewStudent
from utilityFunctions import ensurelogFile

#LOG_DIRECTORY="USER_LOGS"
BORDER="-"*60
#####################################################################################################    
#   Function Name   :   setSessionUserDetails
#   Input Params    :   userName
#   Output Params   :   None
#   Description     :   Set Session User details
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def setSessionUserDetails(userName,studentRecord,studEmail):
    st.session_state["logged_in"]= True
    st.session_state["student_Name"] = userName
    st.session_state["student_info"]=studentRecord
    st.session_state["student_email"]=studEmail
    st.session_state["login_toast"] = {
                    "text": f"🎉 Logged in successfully {userName}!",
                    "icon": "✅"
                }
    #st.balloons()
    #print(st.session_state["student_Name"])
    #st.rerun()
#####################################################################################################    
#   Function Name   :   loadLoginPage
#   Input Params    :   studentDF
#   Output Params   :   None
#   Description     :   This method login in student to the app
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def loadLoginPage(studentDF):
    st.subheader("🔐 Account Login")
    tab1, tab2 = st.tabs(["🔒 Log In", "📝 Register/New User"])
    with tab1:
        #   Login Form
        with st.form("loginForm"):
            studId = st.text_input("User Id", placeholder="Enter your ID")
            studPassword = st.text_input("Password", type="password", placeholder="Enter your password")
            
            userName=st.session_state["student_Name"]
            
            logIn = st.form_submit_button("Sign In",
                                          on_click=createLogFile,
                                          args=(studId,studentDF,))
            if logIn:
                #   Find student record in dataframe
                studentRecord = studentDF[studentDF['studentId'].astype(str).str.strip() == studId.strip()]
                
                if not studentRecord.empty and \
                    str(studentRecord.iloc[0]['password']) == studPassword.strip():
                    studName = studentRecord.iloc[0]['Name']
                    studEmail=studentRecord.iloc[0]['emailId']
                    # set Session User Details
                    setSessionUserDetails(studName,studentRecord,studEmail)   
                    
                else:
                    st.error("❌ Invalid Username or Password.")           
    with tab2:
        with st.form("registerationForm", clear_on_submit=True): 
            regStudId = st.text_input("User Id", placeholder="User Id")
            regStudName = st.text_input("User Name", placeholder="Enter your Name").strip()
            regStudPassword = st.text_input("Password", type="password", placeholder="Password")
            regStudEmail=st.text_input("EmailId",placeholder="Enter EMail Address").strip()
            register = st.form_submit_button("Register Student", use_container_width=True)
            if register:
                if not regStudName or not regStudPassword:
                    st.error("Please fill in both Password and Student Name!")
                else:
                    registerNewStudent(regStudId,regStudName,regStudPassword,regStudEmail) 
    userName=st.session_state["student_Name"]
    st.session_state.page = "start_screen"
    #ensurelogFile(userName)    
    
#####################################################################################################    
#   Function Name   :   createLogFile
#   Input Params    :   studentId,studentD
#   Output Params   :   None
#   Description     :   This method login in student to the app
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def createLogFile(studentId,studentDF):    

    #studId = studentDF[studentDF["studentId"] == studentId]
    studentRecord = studentDF[studentDF['studentId'].astype(str).str.strip() == studentId.strip()]

    print(f"Student Record...{studentRecord}")
    studentFound=not studentRecord.empty
    if not studentRecord.empty:
        studDetails = studentRecord.iloc[0]
        studName=studDetails["Name"]
        st.session_state["student_Name"] = studName
        
        ensurelogFile(studName)    