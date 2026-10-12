"""----------------------------------------------------------------------------------
    Problem Statement   :   Build a chatbot using LangGraph and Streamlit
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
import streamlit as st
from langGraphBackend import chatBot
from langchain.messages import HumanMessage
import uuid
#CONFIG = {'configurable': {'thread_id': 'thread-1'}}

###############################################################################
#   Function        :   generateThreadId
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Generate Thread Id
#   Author          :   Vaishali M Jorwekar
###############################################################################
def generateThreadId():
    thread_id=uuid.uuid4()
    return thread_id
###############################################################################
#   Function        :   addThreads
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Add Threads
#   Author          :   Vaishali M Jorwekar
###############################################################################
def addThreads(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
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
    if 'thread_id' not in st.session_state:
        st.session_state['thread_id']=generateThreadId()
    
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads']=[]
        
    addThreads(st.session_state['thread_id'])        
    
    
    
###############################################################################
#   Function        :   generateStreamData
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Generate Streaming Data
#   Author          :   Vaishali M Jorwekar
###############################################################################
def generateStreamData(userInputText):  
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    streamData= chatBot.stream(
                {
                    'messages':[HumanMessage(content=userInputText)]},
                    config=CONFIG,
                    stream_mode='messages'
                )
    for chunks,metadata in streamData:
        yield chunks.content 
##############################################################################
#   Function        :   resetConversation
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Reset Conversation
#   Author          :   Vaishali M Jorwekar
###############################################################################
def resetConversation(): 
       thread_id=generateThreadId()
       st.session_state['thread_id']=thread_id 
       #addThreads(st.session_state['thread_id'])  
       st.session_state['messageHistory']=[] 
##############################################################################
#   Function        :   loadConversations
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Load Previous Conversations
#   Author          :   Vaishali M Jorwekar
###############################################################################
def loadConversations(threadId): 
       state=chatBot.get_state(config={"configurable":{'thread_id':threadId}})  
       return state.values.get('messages', [])  
###############################################################################
#   Function        :   configureUI
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Configures User Interface
#   Author          :   Vaishali M Jorwekar
###############################################################################
def configureUI():  
    st.sidebar.title("Chatbot using Langgraph")   
    
    if st.sidebar.button("New Conversation"):
        resetConversation()
        
        
    st.sidebar.header("Chat Message History")
    for threadId in st.session_state['chat_threads']:
        if st.sidebar.button(str(threadId)): 
            st.session_state['thread_id']=threadId
            chatMessageHistory=loadConversations(threadId) 
            
            tempMessages=[]
            for msg in chatMessageHistory:
                if isinstance(msg,HumanMessage):
                    role='user'
                else:
                    role='assistant'
                tempMessages.append({'role':role,'content':msg.content})
            st.session_state['messageHistory']=tempMessages                        
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    configureData()
    configureUI()
    
    
    for message in st.session_state['messageHistory']:
        with st.chat_message(message['role']):
            st.text(message['content'])    
    userInputText=st.chat_input("Type Here")
    
    if userInputText:
        st.session_state['messageHistory'].append(
                            {'role': 'user', 
                             'content': userInputText
                             })
        with st.chat_message('user'):
            st.text(userInputText)
            
         
        #CONFIG={'configurable':{'thread_id':st.session_state['thread_id']}}
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