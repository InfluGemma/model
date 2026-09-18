import torch
#import langchain
#from langchain.llms.base import LLM
from typing import Optional, List, TypedDict, Any
#from langchain.agents import initialize_agent, AgentType
#from langchain.memory import ConversationBufferMemory
#from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from huggingface_hub import login
import re
from peft import PeftModel
from tools import tools

model_id = "google/gemma-3-1b-pt"
model_class = AutoModelForCausalLM
model_path = "/srv/scratch/z5397970/influgemma_v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = model_class.from_pretrained(model_id)
base_model.resize_token_embeddings(len(tokenizer))
base_model = base_model.to(torch.float32)
model = PeftModel.from_pretrained(base_model, model_path)
model.config.use_cache = True
model.gradient_checkpointing_disable()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False
    )


llm = HuggingFacePipeline(pipeline=pipe)

react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful flu forecasting agent.\n"
     "Use the tools provided to gather information and reason step by step.\n"
     "Format your responses as:\n\n"
     "Thought: <your reasoning>\n"
     "Action: <tool name or 'final'>\n"
     "Action Input: <input to the tool>\n"
     "Final Answer: <your final output if Action is final>\n"),
    ("human", "{query}")
    ])

#memory = ConversationBufferMemory(
#    memory_key="chat_history",
#    return_messages=True,
#    )

class State(TypedDict):
    messages: List[Any]

def call_llm(state: State):
    query = state["messages"][-1].content
    formatted = react_prompt.invoke({"query":query})
    result = llm.invoke(formatted)
    ai_msg = AIMessage(content=result)
    return {"messages": state["messages"] + [ai_msg]}

def router(state: State):
    last = state["messages"][-1]
    content = last.content

    tool_names = [t.name for t in tools]

    pattern = r"Action:\s*(" + "|".join(map(re.escape, tool_names)) + r")"
    action = re.search(pattern, content)
   # action_input = re.search(r"Action Input:\s*(.*)",content)

    if not action:
        return "end"

    action_name = action.group(1).strip()
   # action_input = action_input.group(1).strip() if action_input else ""
    

    if action_name == "final":
        return "end"

    return action_name

tools = tools

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", router, {
    "Get Current Flu Cases":"tools",
    "Get Past Flu Cases":"tools",
    "Get State Demographic Data":"tools",
    "Get Vaccination Percent":"tools",
    "Get Google Trends":"tools",
    "Get ARIMA Forecast":"tools",
    "Get LSTM Forecast":"tools",
    "Get RNN Embeddings":"tools",
    "Get Reddit Post Percent":"tools",
    "final":"__end__",
    "end":"__end__",
    })

graph.add_edge("tools", "llm")

app = graph.compile()
    


state = "NSW"
prompt = """Forecast flu cases in {state} and predict the trend for cases over the next two weeks from the following options: Substantial Increase, Increase, Stable, Decrease, Substantial Decrease."""

response = app.invoke({"messages": [HumanMessage(content=prompt)]})
print(response["messages"][-1].content)
