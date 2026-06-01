"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    studentDetails.py
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Mock Interview :    studentDetails.py
--------------------------------------------------------------------------------------------------------"""
import streamlit as st
import pandas as pd

#####################################################################################################
STUDENT_DATABASE="projectData/Database/studentData.csv"
BORDER="-"*60
#####################################################################################################    
#   Function Name   :   setStudentDataframe
#   Input Params    :   studentDF
#   Output Params   :   None
#   Description     :   Loads Question Bank from database
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def setStudentDataframe(studentDF):
    st.session_state["All_Student_info"] = studentDF
    #st.rerun()
#####################################################################################################    
#   Function Name   :   registerNewStudent
#   Input Params    :   regStudId,regStudName,regStudPassword
#   Output Params   :   None
#   Description     :   This method adds new Students Data
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def registerNewStudent(regStudId,regStudName,regStudPassword,regStudEmail):
    try:
        df = loadStudentsDetails()
        new_student = {"studentId": str(regStudId).strip(), 
                       "password": str(regStudPassword).strip(),
                       "Name":str(regStudName).strip(),
                       "emailId":str(regStudEmail).strip()}
                
        new_row = pd.DataFrame([new_student])
        updated_df = pd.concat([df,new_row], ignore_index=True)


        # Write Data to csv
        updated_df.to_csv(STUDENT_DATABASE, index=False)
        st.write(f"{STUDENT_DATABASE} File updated and saved ...!")
        
        #print(updated_df)
        
        st.session_state["All_Student_info"] = updated_df
        setStudentDataframe(updated_df)
        
        
        st.session_state["login_toast"] = {
                    "text": f"🎉 Registration successful !!",
                    "icon": "✅"}
        st.rerun()
    except FileNotFoundError: 
        st.error(f"Student Database file not found.")
        st.stop()
#####################################################################################################    
#   Function Name   :   loadStudentsDetails
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This method Loads Students Data
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def loadStudentsDetails():
    try:
       return pd.read_csv(STUDENT_DATABASE,sep=",")  
    except FileNotFoundError: 
        st.error(f"Student Database file not found.")
        st.stop()
#####################################################################################################        
        