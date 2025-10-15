import gc
import os
from typing import TypedDict, Annotated
import operator
from memory_profiler import profile
from langchain.chat_models import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

llm = AzureChatOpenAI(  # Or ChatOpenAI for non-Azure
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    temperature=0.0,
    streaming=False
)

class State(TypedDict):
    messages: Annotated[list, operator.add] 


def joke_node(state: State) -> State:
    topic = state["messages"][-1].content if state["messages"] else "elephants" 
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")
    chain = prompt | llm
    result = chain.invoke({"topic": topic})
    return {"messages": [result]}  # Add to state


workflow = StateGraph(State)
workflow.add_node("joke", joke_node)
workflow.set_entry_point("joke")
workflow.add_edge("joke", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

@profile
def run_graph(topic="elephants", thread_id="test"):
    config = {"configurable": {"thread_id": thread_id}}  # Isolate per thread
    initial_state = {"messages": [HumanMessage(content=topic)]}
    result = graph.invoke(initial_state, config=config)
    print(result["messages"][-1].content)  # Print joke
    del result  # Explicit delete
    gc.collect()  # Force GC

if __name__ == "__main__":
    for i in range(5):  # Loop to check accumulation
        run_graph(f"elephants {i}", thread_id=f"thread_{i}")  # Unique threads to avoid cross-leak