"""-----------------------------------------------------------------------------------------------------
                        Mock Interview :    Mock Test 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement   :  Mock Interview :    Mock Test 
--------------------------------------------------------------------------------------------------------"""
 #Import ask_llm function
# Used to communicate with Llama3 model

from llm_engine import ask_llm

import streamlit as st

BORDER="-"*60

####################################################################################################    
#   Function Name   :   evaluteAnswer
#   Input Params    :   topic,question,answer,id
#   Output Params   :   None
#   Description     :   Sends answer to LLM and evalutes 
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def evaluteAnswer(topic,question,id):
    #userAnswer = f"text_box_input_{id}"
    
    logging=st.session_state["current_logger"]

    #answerText=st.session_state.get(answer, "")
    answerText=""
    audioTextKey = f"text_box_input_{id}"
    textBoxInput = f"text_{id}"

    # Check which key actually exists in Streamlit session state
    if audioTextKey in st.session_state and st.session_state[audioTextKey]:
        answerText = st.session_state[audioTextKey]
    elif textBoxInput in st.session_state and st.session_state[textBoxInput]:
        answerText = st.session_state[textBoxInput]
    else:
        answerText = st.session_state.audio_text_files[id]
        
        
    print(f"-----Audio Text:{st.session_state.get(audioTextKey, "")}") 
    print(f"-----Text:{st.session_state.get(audioTextKey, "")}")  
    print(f"-----{st.session_state.audio_text_files[id]}")
   
    print(f"")
    st.session_state["total_que_answered"]+=1    
    #currentText = st.session_state.audio_text_files.get(id, "")
    #if answerText != currentText:
    #    answerText=currentText
    if "current_logger" in st.session_state:
        logging = st.session_state["current_logger"]
        logging.info(f"Question {id+1} : {question}")
        logging.info(f"Answer: {answerText}")
        logging.info(f"\n{BORDER}\n")    
    print(f"topic:{topic}")
    print(f"Question:{question}")
    print(f"Answer:{answerText}")
    student_name=st.session_state["student_Name"]
    print(f"Student Name:{student_name}")
    
        
    prompt=generatePrompt(topic,question,answerText,student_name)
        
    
    print("Prompt Generated Successfully.")
    print("Sending Prompt to LLM...\n")

        # Send prompt to LLM
    #st.write_stream(ask_llm(prompt))
    
    evaluation = ask_llm(prompt)
    scorePerQuestion=0
    scorePerQuestion=extract_score(evaluation)
    print("scorePerQuestion",scorePerQuestion) 
    print(f"total Marks:{st.session_state["total_Marks"]}")    
    st.session_state[f"eval_output_{id}"] = evaluation
        
    st.session_state["total_Marks"]+=scorePerQuestion
        
    logging.info("Evaluation LLM")
    logging.info(f"{evaluation}")
    logging.info(f"\n{BORDER}")
    logging.info(f"\n{BORDER}")



####################################################################################################    
#   Function Name   :   extract_score
#   Input Params    :   evaluation
#   Output Params   :   None
#   Description     :   Extract score from received response
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
    
def extract_score(evaluation):
        """
        Function Name : extract_score

        Description   :
        Extracts numerical score from LLM response.

        Example:
            Score: 7/10

        Output:
            7

        Parameters    :
            evaluation : LLM generated evaluation

        Return Value :
            Returns extracted score
        """

        try:
            # Split evaluation into multiple lines
            for line in evaluation.split("\n"):

                # Search for line containing score
                if "score" in line.lower():

                    # Replace special characters
                    words = line.replace("/", " ") \
                                .replace(":", " ") \
                                .split()

                    # Traverse words
                    for word in words:

                        # Check whether word is number
                        if word.isdigit():

                            score = int(word)

                            # Validate score range
                            if 0 <= score <= 10:
                                return score

            # Default score if extraction fails
            return 0

        except Exception as e:

            print("\nError while extracting score.")
            print("Error :", e)

            return 0         
####################################################################################################    
#   Function Name   :   generatePrompt
#   Input Params    :   topic,question,answer,student_name
#   Output Params   :   Response From LLM
#   Description     :   Genrate prompt for LLM
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
    
def generatePrompt(topic,question,answer,student_name):
    prompt = f"""
    You are an expert technical interviewer from Marvellous Infosystems.

    Evaluate the student's answer carefully.

    Student Name: {student_name}
    Topic: {topic}
    Question: {question}
    Student Answer: {answer}

    Give response strictly in this format:

    Score: <number out of 10>

    Evaluation:
    Write whether the answer is correct, partially correct, or incorrect.

    Missing Points:
    Mention important technical points missing in the answer.

    Improved Answer:
    Write a better answer in simple student-friendly language.

    Interview Suggestion:
    Give one practical suggestion to improve interview performance.
    """
    return prompt  