"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    Mock Test 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Marvellous SmartHire - AI Mock Interview Agent 
                    •  Develope an AI-powered Mock Interview System capable of conducting 
                       technical interviews and evaluating candidate answers intelligently.
--------------------------------------------------------------------------------------------------------"""
from backgroundSettings import setbackground
import streamlit as st
import logging
import os,random
from audioInput import recordAndSaveAudioWithStreamlit
from utilityFunctions import LOG_DIRECTORY
import speech_recognition as sr
from interviewAgent import evaluteAnswer    
from generateReport import generateAndSendReport 
from utilityFunctions import initializeSessionVariables,clearSessionCache


BORDER="-"*60
#####################################################################################################    
#   Function Name   :   pageSettings
#   Input Params    :   userName,questionBank
#   Output Params   :   None
#   Description     :   Mock Test Page Settings
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################
def pageSettings(userName):
    with st.sidebar:
        setbackground(imgWidth=330)

        sb_col1, sb_col2 = st.columns(2)
        with sb_col1:
                # Displays username as a subheader 
                
                
                st.write(f"👤 **{userName}**")
                
        with sb_col2:
                # Log out button
                if st.button("Log out",
                        key="logout_button",
                        #on_click=initializeSessionVariables, 
                        type="secondary",
                        use_container_width=True) : 
                    clearSessionCache()
                    initializeSessionVariables()
                    
                    st.session_state.page = "login_screen" 
                    st.session_state["logged_in"] = False
                    st.rerun()
#####################################################################################################    
#   Function Name   :   startMockTest
#   Input Params    :   userName,questionBank
#   Output Params   :   None
#   Description     :   Mock Test Logic
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################

def startMockTest(userName):
    
    #logger.info("Mock Test Started")
    pageSettings(userName)
    st.subheader("Technical Mock Interview")
    if st.session_state.page == "start_screen":
        
        startMockTestPage()
        #st.rerun() 
    elif st.session_state.page == "quiz_screen":
        st.subheader("📝 Full Mock Test Started")
        #st.session_state.exam_Type="Full Mock Test"
        #fullMockTest()
        # If Full Mock Test Load all questions
        questionBank=st.session_state.get("loaded_questions",[])
        if len(questionBank)>5:
            questionBank=random.sample(questionBank,5)
        for i in range(0,len(questionBank)):
            st.session_state.setdefault(f"show_recorder_{i}", False)
        # Load Test questions
        loadTestQuestions(questionBank)
        
    elif st.session_state.page == "topic_quiz_screen": 
        questionBank=st.session_state.get("loaded_questions",[])
   
        topics = list(set([q[1] for q in questionBank]))

        topicSelected= st.selectbox(f"Select Topic:",topics)

        st.session_state['topicSelected']=topicSelected
        
        topicSelected=st.session_state['topicSelected'] 
        
        #st.session_state.exam_Type=f"Topic Wise : {topicSelected}"
        
        filtered_questions = [que for que in questionBank if que[1] == topicSelected]
        if len(filtered_questions)>5:
            filtered_questions=random.sample(filtered_questions,5)
        for i in range(0,len(filtered_questions)):
            st.session_state.setdefault(f"show_recorder_{i}", False)    
        # Load Test questions
        loadTestQuestions(filtered_questions)

    
            
####################################################################################################    
#   Function Name   :   startMockTestPage
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   UI Logic for test type
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################  
def startMockTestPage():
    
    with st.form("MockTestForm"):
        startMockTest = st.form_submit_button("Start Full Mock Test")
        topicWiseMockTest = st.form_submit_button("Topic Wise Mock Test")

        questionBank=st.session_state.get("loaded_questions",[])
            
        if startMockTest:
                #questionBank=st.session_state.get("loaded_questions",[])
            if questionBank:
                st.write(questionBank)
                st.session_state.page = "quiz_screen"
                st.session_state.exam_Type="Full Mock Test"
                st.rerun() 
        if topicWiseMockTest:
            questionBank=st.session_state.get("loaded_questions",[])
            if questionBank:
                
                st.session_state.page = "topic_quiz_screen"
                st.session_state.exam_Type="Topic Wise Mock Test"

                st.rerun()    
        if "current_logger" in st.session_state:
            logging = st.session_state["current_logger"]
            logging.info(f"\n      Exam Type    :   {st.session_state['exam_Type']}")
            
            if "topicSelected" in st.session_state:
                logging.info(f"\n      Topic    :   {st.session_state['topicSelected']}")
                 
            logging.info(f"\n{BORDER}")              
####################################################################################################    
#   Function Name   :   loadTestQuestions
#   Input Params    :   questionBank
#   Output Params   :   None
#   Description     :   UI and Logic for  Mock Test 
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 

def loadTestQuestions(questionBank):
   with st.container(border=True):
        st.caption(f"Question {st.session_state.current_page+1} of {len(questionBank)}")

        currentPageIndex = st.session_state.current_page
        if 0 <= currentPageIndex < len(questionBank):
            current_question = questionBank[currentPageIndex]
            # Load Mock test page
            questionBlock(currentPageIndex, current_question, questionBank)
                
        #print(f"currentPageIndex.....{currentPageIndex}") 
        if (currentPageIndex == len(questionBank) - 1) and \
            st.button("End Test",use_container_width=True):
                #   Generate and send report
                generateAndSendReport()
                #st.session_state.page = "start_screen"
                
                initializeSessionVariables()
                clearSessionCache()
                #st.session_state.clear()
        
                #st.session_state["logged_in"] = False
                st.session_state.page = "login_screen"     
                st.session_state.logged_in = False 
                st.rerun()
                
            
            
####################################################################################################    
#   Function Name   :   questionBlock
#   Input Params    :   queId, question, questionBank
#   Output Params   :   None
#   Description     :   UI and Logic for  Mock Test 
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
@st.fragment            
def questionBlock(queId, question, questionBank):           
    st.write(f"Question {queId + 1}: {question[3]}") 
    show_recorder_key = f"show_recorder_{queId}"
    if show_recorder_key not in st.session_state:
        st.session_state[show_recorder_key] = False               
    with st.container(border=True):
        #################################################
        #if not st.session_state[show_recorder_key]:

        #    st.button(
        #           label=f"🎙️ Record Answer : {queId + 1}",
        #           key=f"button_rec_{queId}",
        #            on_click=enableRecording,
        #            args=(queId,question[3]),
        #            use_container_width=True
        #            )
                            
        #else: 
        #####################Un comment this############################  
                 
            recordAndSaveAudioWithStreamlit(queId,question[3])
            prev_mic_key = f"microphone_Record_{queId - 1}"
            if prev_mic_key in st.session_state:
                del st.session_state[prev_mic_key] 
                
    if "audio_files" in st.session_state:
         hasAudioFile = queId in st.session_state["audio_files"]
    else:
        hasAudioFile = False                  
    #print(f"----------Audio---{hasAudioFile}")
    with st.container(border=True):         
                #col2,col3 = st.columns(2)     
                #with col2:  
                
                #print(f"File Id:{st.session_state.audio_files[id]}")
        st.button(label=f"📝Answer In Text : {queId + 1}",
                    key=f"button_Ans_ToText_{queId}",
                    on_click=convertAnswerToText,
                    args=(queId,question[3]),
                    disabled=not hasAudioFile
                ) 
        msg_type=""
        msg_text=""
        error_info = st.session_state.get("ui_errors", {}).get(queId)
        if error_info:
            msg_type, msg_text = error_info
        if msg_type == "error":
            st.error(msg_text)
        elif msg_type == "warning":
             st.warning(msg_text)
             
             
        if queId in st.session_state.audio_text_files and hasAudioFile:     
            st.markdown("**Audio To Text Conversion:**")    
            st.text_area(
                            label=f"Edit Answer {queId+1}", 
                        height=100,
                            #value=st.session_state.audio_text_files[id],
                        key=f"text_box_input_{queId}",
                        on_change=update_text,
                        args=(queId,)

                      )
            
    with st.container(border=True):         
        current_text = st.session_state.audio_text_files.get(queId, "")
        has_text_input = bool(current_text.strip())               
        print(f"has_text_input:{has_text_input}")
                    #with col3:
        st.button(label=f"📊 Evaluate Answer:{queId+1}",
                 key=f"Evaluate_{queId}",
                 on_click=evaluteAnswer,
                 args=(question[1], question[3],queId),
                 disabled=not has_text_input)
                    
        eval_result_key = f"eval_output_{queId}"
        if eval_result_key in st.session_state:
            with st.container(border=True):
                st.write("📊 **Evaluation Result:**")
                        #st.success(st.session_state[eval_result_key]) 
                st.success(st.session_state[eval_result_key])    
   
    
        st.markdown("---")
        if st.button(
                "Next Question ➡️", 
                #on_click=nextQuestion, 
                #args=(questionBank,),
                disabled=(st.session_state.current_page == len(questionBank)-1),
                use_container_width=True
            ):
            if st.session_state.current_page < len(questionBank) - 1:
                st.session_state.current_page += 1
                st.rerun(scope="app")
            
####################################################################################################    
#   Function Name   :   enableRecording
#   Input Params    :   questionIndex,questionText
#   Output Params   :   None
#   Description     :   Convert Audio to text
#   Author          :   Vaishali M. Jorwekar              
#######################################################################################             
def enableRecording(questionIndex,questionText):
    show_recorder_key = f"show_recorder_{questionIndex}"
    st.session_state[show_recorder_key] = True         
####################################################################################################    
#   Function Name   :   update_text
#   Input Params    :   question_id
#   Output Params   :   None
#   Description     :   Convert Audio to text
#   Author          :   Vaishali M. Jorwekar              
#######################################################################################     
def update_text(question_id):
    #key = f"text_box_input_{id}"
    #if key in st.session_state:
     #   st.session_state.audio_text_files[id] = st.session_state[key]
    typed_text = st.session_state[f"text_box_input_{question_id}"]
    st.session_state.audio_text_files[question_id] = typed_text

####################################################################################################    
#   Function Name   :   nextQuestion
#   Input Params    :   questionBank
#   Output Params   :   None
#   Description     :   Convert Audio to text
#   Author          :   Vaishali M. Jorwekar              
#######################################################################################     
def nextQuestion(questionBank):
    if st.session_state.current_page < len(questionBank) - 1:
        st.session_state.current_page += 1
    #st.rerun(scope="app")              
####################################################################################################    
#   Function Name   :   convertAnswerToText
#   Input Params    :   audio_file
#   Output Params   :   None
#   Description     :   Convert Audio to text
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def convertAnswerToText(q_id,question):
    
    if "ui_errors" not in st.session_state:
        st.session_state.ui_errors = {}
        
    # Clear previous error for this specific question
    st.session_state.ui_errors[q_id] = None
    
    
    #audio_file=st.session_state["current_audio_file"]
    if "audio_files" not in st.session_state or \
        q_id not in st.session_state.audio_files:
        #st.error("No recording found for this question.")
        st.session_state.ui_errors[q_id] = ("error", "No recording found.")
        return
        
    audio_file = st.session_state.audio_files[q_id]
  
    print(f"convertAnswerToText:{audio_file}   {q_id}")
  
    if q_id in st.session_state.audio_files:
        if not audio_file or not os.path.exists(audio_file):
            #st.error(f"Target audio file does not exist: {audio_file}")
            st.session_state.ui_errors[q_id] = ("error", "Audio file empty or missing.")
            return

        if os.path.getsize(audio_file) == 0:
            #st.warning("The recorded audio file is empty. Please check your mic input.")
            st.session_state.ui_errors[q_id] = ("warning", "Audio file empty or missing.")

            return
        recognizer = sr.Recognizer()
        try:
            #audio_file.seek(0)
            with sr.AudioFile(audio_file) as audio_source:
                audio_data = recognizer.record(audio_source)
        
                text = recognizer.recognize_google(audio_data)
                st.session_state.audio_text_files[q_id] = text  
                #st.session_state["Answer_Text"]=text
                st.session_state[f"text_box_input_{q_id}"] = text
                #st.session_state[f"audio_text_files_{q_id}"]=text
                #if "current_logger" in st.session_state:
                 #   logging = st.session_state["current_logger"]
                 #   logging.info(f"Answer: {text}")
                            
        except sr.UnknownValueError:
                #st.warning("Audio was unclear or silent.")
                st.session_state.ui_errors[q_id] = ("warning", "Audio was unclear.")

        except sr.RequestError:
                #st.error("API connection issues with Google Web Speech.")
                st.session_state.ui_errors[q_id] = ("error", "API connection issue.")

        except NameError:
                    pass  
        #except FileNotFoundError:
        #    st.error(f"Could not find '{audio_file}'. Please place it in the same folder.")
        #st.rerun()                                 
      
