"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    Audio Input
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Mock Interview :    Audio Input
--------------------------------------------------------------------------------------------------------"""
import sounddevice as sd
import os
from scipy.io.wavfile import write
import scipy.io.wavfile as wav
import streamlit as st

from utilityFunctions import ensure_dir,formatTime
import numpy as np
from utilityFunctions import LOG_DIRECTORY
import datetime
#MAIN_DIR="ProjectData"


from streamlit_mic_recorder import mic_recorder


SUB_DIR="Answer"
FILENAME="answer"


####################################################################################################    
#   Function Name   :   getAudioFilePath
#   Input Params    :   questionIndex,question
#   Output Params   :   Recoreded and saved file
#   Description     :   Recording audio with streamlit
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################  
def getAudioFilePath(questionIndex,question):
    
    
    userName=st.session_state["student_Name"]
    
    userDir=os.path.join(LOG_DIRECTORY,userName)
    formatDate=datetime.datetime.now().strftime("%d_%m_%Y")
    userDir=os.path.join(userDir,formatDate)
    dirPath=os.path.join(userDir,SUB_DIR)
    print(dirPath)
    ensure_dir(dirPath)
    fileName=f"{FILENAME}_{questionIndex+1}_at_{formatTime()}.wav"
    filePath=os.path.join(dirPath,fileName)
    print(f"Audio Is saved at:{filePath}")
    return filePath
####################################################################################################    
#   Function Name   :   recordAndSaveAudioWithStreamlit
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Recording audio with streamlit
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################  
def recordAndSaveAudioWithStreamlit(questionIndex,question):
    microphoneKey = f"microphone_Record_{questionIndex}"
    lock_key = f"file_written_{questionIndex}"
    audio_file = st.audio_input(
        "Click 🎙️ to record", 
        width="stretch", 
        key=microphoneKey
    )
    
    #################################################################
    use_text = st.checkbox("⌨️ Type answer instead", key=f"toggle_{questionIndex}")

    if use_text:
            # Text Field Option
            user_text = st.text_area("Type your answer here:", key=f"text_{questionIndex}")
            if user_text:
                #st.session_state.audio_files[questionIndex] = user_text
                st.session_state.audio_text_files[questionIndex]=user_text
                #st.session_state["total_que_answered"] += 1

    ##########################################################################
    elif audio_file is not None:
        # Playback the current recording
        st.audio(audio_file)
        audio_Filepath = getAudioFilePath(questionIndex, question)

        if not st.session_state.get(lock_key, False) and \
            (questionIndex not in st.session_state.audio_files):
            try:
                # Save the binary data
                with open(audio_Filepath, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                st.session_state.audio_files[questionIndex] = audio_Filepath  
                st.session_state["current_audio_file"] = audio_Filepath 
                st.session_state[lock_key] = True  
            
                if questionIndex not in st.session_state["recorded_questions_log"]:
                    #st.session_state["total_que_answered"] += 1
                    st.session_state["recorded_questions_log"].add(questionIndex)               
                
                if hasattr(audio_file, 'close'):
                    audio_file.close()
                            
                st.success(f"💾 Audio saved successfully to: `{audio_Filepath}`")
                st.rerun(scope="fragment")
                
                
            except Exception as e:
                    # Capture writing failures without locking the system execution
                    st.error(f"Failed to process or save audio : {str(e)}")                
                #st.rerun()

#####################################################################################################    
#   Function Name   :   generateFilePath
#   Input Params    :   questionIndex
#   Output Params   :   None
#   Description     :   File path generation
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def generateFilePath(questionIndex,userDir):
    dirPath=os.path.join(userDir,SUB_DIR)
    print(dirPath)
    ensure_dir(dirPath)
    fileName=f"{FILENAME}_{questionIndex+1}_at_{formatTime()}.wav"
    filePath=os.path.join(dirPath,fileName)
    return filePath
             
#####################################################################################################    

