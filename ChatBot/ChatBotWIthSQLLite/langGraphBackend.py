"""----------------------------------------------------------------------------------
    Problem Statement   :   Build a chatbot using LangGraph and Stremlit
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""

from langgraph.graph import START,END,StateGraph
from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.messages import HumanMessage

import sqlite3
###############################################################################
#   Function        :   ChatState
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   ChatState 
#   Author          :   Vaishali M Jorwekar
###############################################################################
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
###############################################################################
#   Function        :   ChatNode
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   ChatNode 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def ChatNode(state:ChatState):
    messages=state["messages"]
    llm = ChatOllama(model="llama3", temperature=0)
    response=llm.invoke(messages)
    return {"messages":[response]}   
   

###################################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###########################################################################
def buildChatGraph():
    # SqlLite Database
    #checkpointer = getCheckpointer()
    graph=StateGraph(ChatState)
    graph.add_node("chatNode",ChatNode)
    graph.add_edge(START,"chatNode")
    graph.add_edge("chatNode",END)
    
    return graph.compile(checkpointer=checkpointer)
###################################################################################
#   Function        :   getCheckpointer
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Get checkpointer for SqlLite Database
#   Author          :   Vaishali M Jorwekar
###########################################################################
def getCheckpointer():
    connection=sqlite3.connect(database="ChatBotDatabase.db",check_same_thread=False)
    checkpointer=SqliteSaver(conn=connection)
    return checkpointer

###################################################################################
#   Function        :   getAllThreads
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Retrieves all the threads from the database
#   Author          :   Vaishali M Jorwekar
###########################################################################
def getAllThreads():
    
    allThreads=set()
    for checkpoint in checkpointer.list(None):
        allThreads.add(checkpoint.config["configurable"]["thread_id"])
    return list(allThreads)
###############################################################################    
checkpointer=getCheckpointer()
chatBot=buildChatGraph()
getAllThreads()
CONFIG = {'configurable': {'thread_id': 'thread-2'}}
response= chatBot.invoke(
                {
                    'messages':[HumanMessage(content="Hi,What is ML")]},
                    config=CONFIG,
                )
print(f"response : {response}")
###############################################################################

    