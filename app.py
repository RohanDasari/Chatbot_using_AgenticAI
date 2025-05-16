# import streamlit as st
# from transformers import pipeline
# from bot import chatbot
# from langchain_core.messages import HumanMessage

# mybot=chatbot()
# workflow=mybot()
# config = {"configurable": {"thread_id": "1"}}
# # Set up the Streamlit app UI
# st.title("ChatBot with LangGraph")
# st.write("Ask any question, and I'll try to answer it!")

# # Input text box for the question
# question = st.text_input("Enter your question here:")
# input = question
# # input={"messages": [question]}

# # Button to get the answer
# if st.button("Get Answer"):
#     if input:
        
#         response = workflow.invoke({"messages": [HumanMessage(content=input)]}, config)
#         st.write("**Answer:**", response['messages'][-1].content)
#     else:
#         st.warning("Please enter a question to get an answer.")

# # Additional styling (optional)
# st.markdown("---")
# st.caption("Powered by Streamlit and Transformers")

# import streamlit as st
# from langchain_core.messages import HumanMessage
# from bot import chatbot
# import uuid

# # Initialize chatbot and workflow
# mybot = chatbot()
# workflow = mybot()

# # Initialize thread_id for memory management if not already in session
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())

# # Set up the Streamlit app UI
# st.title("ChatBot with LangGraph")
# st.write("Ask any question, and I'll try to answer it!")

# # Input text box for the question
# question = st.text_input("Enter your question here:")

# # Button to get the answer
# if st.button("Get Answer"):
#     if question:
#         config = {"configurable": {"thread_id": st.session_state.thread_id}}
#         response = workflow.invoke({"messages": [HumanMessage(content=question)]}, config)
#         st.write("**Answer:**", response['messages'][-1].content)
#     else:
#         st.warning("Please enter a question to get an answer.")

# st.markdown("---")
# st.caption("Powered by Streamlit and LangGraph")


# import streamlit as st
# from langchain_core.messages import HumanMessage
# from bot import chatbot
# import uuid

# # Initialize chatbot and workflow
# mybot = chatbot()
# workflow = mybot()

# # Setup session thread_id once per session
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())

# # Setup message history if not already there
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # UI
# st.title("ChatBot with LangGraph + Memory")
# st.write("Chat with memory — tell it something, and refer to it later!")

# user_input = st.text_input("You:")

# if st.button("Send"):
#     if user_input:
#         # Add user message to history
#         st.session_state.chat_history.append(HumanMessage(content=user_input))

#         # Call LangGraph workflow, only inputting latest message
#         config = {"configurable": {"thread_id": st.session_state.thread_id}}
#         response = workflow.invoke({"messages": [st.session_state.chat_history[-1]]}, config)

#         # Add model response to history
#         st.session_state.chat_history.append(response["messages"][-1])

# # Show conversation history
# st.markdown("---")
# for msg in st.session_state.chat_history:
#     if isinstance(msg, HumanMessage):
#         st.markdown(f"**You:** {msg.content}")
#     else:
#         st.markdown(f"**Bot:** {msg.content}")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bot import chatbot
import uuid

# Initialize chatbot and workflow
mybot = chatbot()
workflow = mybot()

chat_history = []
# Setup session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
print(st.session_state.thread_id)
st.title("AgenticAI based Chatbot + soon to be working - MemorySaver")
st.write("This project features an AI assistant powered by LangChain and LangGraph, integrating large language models with custom tools and conditional workflows. It maintains conversation memory, dynamically chooses when to use retrieval tools, and ensures structured, validated responses. The workflow graph can be visualized and exported for easy understanding and debugging.")
st.write("The knowledge base is limited to LLM articles. Whatever queries apart from this will be dynamically scrolled through out the internet and will be used to generate the content")

# Input from user
question = st.text_input("Enter your question here:")
if st.button("Send") and question:
    user_msg = HumanMessage(content=question)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.chat_history.append(HumanMessage(content=question))
    # chat_history.append()
    print("st.session_state.chat_history , ", st.session_state.chat_history)
    # Only pass the current user message
    response = workflow.invoke({'question':question}, config=config)

    print("response ",response)
    # Update session chat history manually for UI (optional, not needed for memory)
    # st.session_state.chat_history.append(user_msg)
    st.session_state.chat_history.append(AIMessage(content=response["generation"].Response))
    print("st.session_state.chat_history ", st.session_state.chat_history)
    print("st.session_state.chat_history question content ", st.session_state.chat_history[0].content)
    # st.markdown(response["generation"].Response)
# Display conversation
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Bot:** {msg.content}")
