"""----------------------------------------------------------------------------------
    Problem Statement   :   Build a chatbot using LangGraph and Stremlit
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
import streamlit as st
from langGraphBackend import chatBot
from langchain.messages import HumanMessage
CONFIG = {'configurable': {'thread_id': 'thread-1'}}
###############################################################################
#   Function        :   configureData
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Configure Initial Parameters
#   Author          :   Vaishali M Jorwekar
###############################################################################
def configureData():
   
    #   If first time then initialise message history
    if 'messageHistory' not in st.session_state:
        st.session_state['messageHistory']=[]
    #   Else append it    
    for message in st.session_state['messageHistory']:
        with st.chat_message(message['role']):
            st.text(message['content'])
###############################################################################
#   Function        :   generateStreamData
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Generate Streaming Data
#   Author          :   Vaishali M Jorwekar
###############################################################################
def generateStreamData(userInputText):  
    streamData= chatBot.stream(
                {
                    'messages':[HumanMessage(content=userInputText)]},
                    config=CONFIG,
                    stream_mode='messages'
                )
    for chunks,metadata in streamData:
        yield chunks.content              
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    configureData()
    userInputText=st.chat_input("Type Here")
    
    if userInputText:
        st.session_state['messageHistory'].append(
                            {'role': 'user', 
                             'content': userInputText
                             })
        with st.chat_message('user'):
            st.text(userInputText)
            
         
        
        with st.chat_message('assistant'):
            aiMessage=st.write_stream(generateStreamData(userInputText))
            st.session_state['messageHistory'].append({
                                                    'role': 'assistant',
                                                    'avatar':"🧑‍💻",
                                                    'content': aiMessage
                                                    }
                                                  )    
            
        

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()