"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    utilityFunctions.py
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Mock Interview :    tilityFunctions.py
--------------------------------------------------------------------------------------------------------"""
import streamlit as st
import pandas as pd
import os,logging,sys
import datetime,json
CONFIG_FILE="ProjectData/Config/config.json"

LOG_DIRECTORY="USER_LOGS"
BORDER="-"*60
#####################################################################################################    
#   Function Name   :   loadConfigFile
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Loads Config file
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def loadConfigFile():
    configFile=json.load(open(CONFIG_FILE))
    return configFile
#####################################################################################################    
#   Function Name   :   initializeSessionVariables
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Initializes Session variables across streamlit application
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def initializeSessionVariables():
    sessionVar = {
        "logged_in": False,
        "student_info": None,
        "student_Name": None,
        "All_Student_info": None,
        "current_logger": None,
        "loaded_questions": None,
        "last_logged_index": -1,
        "logger_initialized": False,
        "current_audio_file": None,
        "page": "login_screen",
        "Answer_Text": "",
        "topicSelected": None,
        "total_que_answered": 0,
        "total_Marks": 0,
        "student_email": None,
        "current_page": 0,
        "exam_Type":None
        }
    for key, val in sessionVar.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = {} 
    if "audio_text_files" not in st.session_state:
        st.session_state.audio_text_files = {} 
    if "recorded_questions_log" not in st.session_state:
        st.session_state["recorded_questions_log"] = set()      
    for q_idx in range(6):
        st.session_state.setdefault(f"file_written_{q_idx}", False)
        st.session_state.setdefault(f"show_recorder_{q_idx}", False)
        st.session_state.setdefault(f"text_box_input_{q_idx}", "")
        st.session_state.setdefault(f"text_{q_idx}","")
        st.session_state.setdefault(f"eval_output_{q_idx}", "")
        st.session_state.setdefault(f"rec_audio_{q_idx}", None)

    if "login_toast" in st.session_state:
        st.toast(
        st.session_state["login_toast"]["text"],
        icon=st.session_state["login_toast"]["icon"]
        )
        del st.session_state["login_toast"] 
    if "topicSelected" in st.session_state:
        del st.session_state["topicSelected"]            
#####################################################################################################    
#   Function Name   :   clearSessionCache
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   InitiaClearslizes Session variables across streamlit application
#   Author          :   Vaishali M. Jorwekar              
####################################################################################################
def clearSessionCache():
    #  Identify all dynamic keys that should be wiped out
    keys_to_delete = []
    
    for key in st.session_state.keys():
        if (
            key.startswith("eval_output_") or 
            key.startswith("text_box_input_") or 
            key.startswith("file_written_") or 
            key.startswith("microphone_Record_") or
            key.startswith("show_recorder_") or 
            key.startswith("text_") or
            key.startswith("toggle_")
        ):
            keys_to_delete.append(key)
            
    #  Safely remove them from the Streamlit memory pool
    for key in keys_to_delete:
        del st.session_state[key]
        
    # Explicitly reset your master tracking dictionaries/sets
    st.session_state.audio_files = {}
    st.session_state.audio_text_files = {}
    st.session_state.current_page = 0
    st.session_state["recorded_questions_log"] = set()
    st.session_state["total_que_answered"] = 0
    st.session_state["total_Marks"]=0
    

###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
############################################################################################
def ensure_dir(path:str):
    try:
        os.makedirs(path,exist_ok=True) 

    except OSError as e:
        print(f"An OS error occurred: {e}")
        exit()
    except Exception as e:
        print(f"An Exception occurred: {e}")
        exit()              

########################################################################################################
#   Function        :   getCurrFormattedTime
#   Input Params    :   None
#   Output Params   :   Current formatted time
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
########################################################################################################
def getCurrFormattedTime():
    currDateTime=datetime.datetime.now()
    currTime=currDateTime.strftime("%I:%M:%S %p")
    return currTime  

########################################################################################################
#   Function        :   formatTime
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
########################################################################################################
def formatTime():
    currTime=getCurrFormattedTime()
    currTime=currTime.replace(":","_")
    currTime=currTime.replace(" ","_")
    return currTime          
###########################################################################################
#   Function        :   ensurelogFile
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   9 Nov 2025
############################################################################################
def ensurelogFile(userName):
    try:
        
        #print("1",userName)
        ensure_dir(LOG_DIRECTORY)
        userDir=os.path.join(LOG_DIRECTORY,userName)
        ensure_dir(userDir)

        userLogFolder=os.path.join(LOG_DIRECTORY,userName)
        ensure_dir(userLogFolder)
        # Date Wise Log Folder
        formatDate=datetime.datetime.now().strftime("%d_%m_%Y")
        userLogFolder=os.path.join(userLogFolder,formatDate)
        ensure_dir(userLogFolder)
        userFileNamePrefix=userName.replace(" ","_")
        
        userLogFile=f"{userFileNamePrefix}_{formatTime()}.log"
        logFilePath=os.path.join(userLogFolder,userLogFile)
        #print(logFilePath)
        # Using formatTime or a unique key prevents reuse of cached loggers
        logger_id = f"{userFileNamePrefix}_{formatTime()}"
        logger = logging.getLogger(logger_id)
        
        # 3. Clean up any leftover handlers on this specific logger instance
        logger.handlers = []
        logger.setLevel(logging.INFO)
        
        # 4. Attach the unique file handler
        file_handler = logging.FileHandler(logFilePath, mode="w", encoding="utf-8")
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
            
        # 5. Save the logger reference to Streamlit's session state so you can use it later
        st.session_state["current_logger"] = logger
        
        
        logger.info(f"{BORDER}")
        logger.info(f"              Mock Test Details of : '{userName}'")
        logger.info(f"{BORDER}")
        currentTime=datetime.datetime.now()
        logger.info(f"\n      Login Details:        ")
        logger.info(f"\n      Date: {currentTime.strftime("%Y-%m-%d")}      ")
        logger.info(f"\n      Time : {currentTime.strftime("%H:%M:%S")}     ")
        logger.info(f"{BORDER}")
    except IOError as e:
        # Code to execute for other I/O errors
        print(f"An I/O error occurred: {e}")
        sys.exit()
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        sys.exit()
    return logFilePath    