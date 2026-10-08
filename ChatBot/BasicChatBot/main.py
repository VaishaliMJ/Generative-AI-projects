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
            
        response= chatBot.invoke({'messages':[HumanMessage(content=userInputText)]},config=CONFIG) 
    
        aiMessage=response['messages'][-1].content 
        st.session_state['messageHistory'].append({
                                                    'role': 'assistant',
                                                    'avatar':"🧑‍💻",
                                                    'content': aiMessage
                                                    }
                                                  )
        with st.chat_message('assistant'):
            st.text(aiMessage)
        

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    
    main()