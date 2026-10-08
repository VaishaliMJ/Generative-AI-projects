"""----------------------------------------------------------------------------------
    Problem Statement   :   Build a chatbot using LangGraph and Stremlit
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""

from langgraph.graph import START,END,StateGraph
from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
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
   

###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def buildChatGraph():
    checkpointer=InMemorySaver()
    graph=StateGraph(ChatState)
    graph.add_node("chatNode",ChatNode)
    graph.add_edge(START,"chatNode")
    graph.add_edge("chatNode",END)
    return graph.compile(checkpointer=checkpointer)
###############################################################################    
chatBot=buildChatGraph()
###############################################################################

    