# # importing a necessary library
# from langgraph.graph import StateGraph,MessagesState, START, END
# from langgraph.graph.message import add_messages
# from typing import Annotated, Literal, TypedDict
# from typing import List
# from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import ToolNode
# from langchain_groq import ChatGroq
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.utilities import GoogleSerperAPIWrapper
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.agents import initialize_agent, Tool
# from langgraph.checkpoint.memory import MemorySaver
# from pydantic import BaseModel, Field
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain.output_parsers import PydanticOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain.schema import Document
# import os
# from dotenv import load_dotenv
# # Load environment variables from a .env file
# load_dotenv()

# # Access environment variables
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"]= GROQ_API_KEY
# os.environ["SERPER_API_KEY"]= SERPER_API_KEY


# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""

#     binary_score: str = Field(
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )
# class RewriteOutput(BaseModel):
#     """Question rewriter to provier improved questions"""
#     questions: list[str] = Field(description="List of two rewritten questions")

# class FinalOutput(BaseModel):
#     """Answer generation to get answer to given question"""
#     Response: str = Field(description="Final response from LLM")

# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.

#     Attributes:
#         question: question
#         generation: LLM generation
#         web_search: whether to add search
#         documents: list of documents
#     """

#     question: str
#     generation: str
#     web_search: str
#     documents: List[str]

# class chatbot:
#     def __init__(self):
#         self.llm=ChatGroq(model_name="deepseek-r1-distill-llama-70b")
#         # self.llm=ChatGroq(model_name="Gemma2-9b-It")
#         self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
#         self.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         self.google_search = GoogleSerperAPIWrapper()
#         self.memory = MemorySaver()
#         self.urls = [
#             "https://lilianweng.github.io/posts/2023-06-23-agent/",
#             "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#             "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
#             ]   
#     # def call_tool(self):
#     #     tool = TavilySearchResults(max_results=2)
#     #     tools = [tool]
#     #     self.tool_node = ToolNode(tools=[tool])
#     #     self.llm_with_tool=self.llm.bind_tools(tools)
#     def create_vectors(self):
#         docs = [WebBaseLoader(url).load() for url in self.urls]
#         docs_list = [item for sublist in docs for item in sublist]

#         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             chunk_size=250, chunk_overlap=25
#         )
#         doc_splits = text_splitter.split_documents(docs_list)

#         # Add to vectorDB
#         vectorstore = Chroma.from_documents(
#             documents=doc_splits,
#             collection_name="rag-chroma",
#             embedding=self.embeddings,
#         )
#         self.retriever = vectorstore.as_retriever()
#         return self.retriever
    
#     def create_grader(self):
#         # Prompt
#         system1 = """You are a grader assessing relevance of a retrieved document to a user question. \n 
#             If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
#             Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
#         grade_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system1),
#                 ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#             ]
#         )

#         self.retrieval_grader = grade_prompt | self.structured_llm_grader
#         return self.retrieval_grader
    
#     def create_rewriter(self):
#         # system2 = """You a question re-writer that converts an input question to a better version that is optimized \n 
#             # for web search. Please give me just 2 optimized questions without any additional text. Please **DO NOT** give me additional information or reasoning."""
#         system2 = """You are a question re-writer that improves a user question for web search.

#         Your task is to generate exactly **two improved versions** of the user's question.

#         ⚠️ Strict rules:
#         - DO NOT return any reasoning, internal thinking, or tags like <think>.
#         - DO NOT include any explanation.
#         - Just return exactly two lines, each containing a rewritten version.

#         🧾 Format (must follow exactly):
#         1. <First improved question>
#         2. <Second improved question>
#         """

#         parser = PydanticOutputParser(pydantic_object=RewriteOutput)
#         prompt = PromptTemplate(
#         template="""You are a question re-writer that improves a user question for web search.

#         Your task is to generate exactly **two improved versions** of the user's question.. Return only JSON. Question: {question}\n{format_instructions}""",
#         input_variables=["question"],
#         partial_variables={"format_instructions": parser.get_format_instructions()}
#         )
#         # re_write_prompt = ChatPromptTemplate.from_messages(
#         #     [
#         #         ("system", system2),
#         #         (
#         #             "human",
#         #             "Here is the initial question: \n\n {question} \n Formulate an improved question.",
#         #         ),
#         #     ]
#         # )

#         self.question_rewriter = prompt | self.llm | parser
#         # self.question_rewriter = " ".join(self.question_rewriter)
#         return self.question_rewriter
    
#     def create_rag_chain(self):
#         # # Prompt
#         # # prompt = hub.pull("rlm/rag-pro÷mpt")
#         # prompt = """You are an assistant for question-answering tasks. 
#         #         Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
#         #         Use five sentences maximum and keep the answer concise.\n
#         #         Question: {question} \nContext: {context} \nAnswer:"""

#         # Post-processing
#         def format_docs(docs):
#             return "\n\n".join(doc.page_content for doc in docs)
#         parser = PydanticOutputParser(pydantic_object=FinalOutput)
#         prompt = PromptTemplate(
#             template="""You are an assistant for question-answering tasks. 
#         Use the following context to answer the question.
#         Keep the answer concise and under 8-9 sentences. 
#         **Do not provide any reasoning or commentary.**
#         STRICTLY FORMATE YOUR RESPONSE AS GIVEN BELOW-
#         {format_instructions}

#         Question: {question}
#         Context: {context}
#         Answer:""",
#             input_variables=["question", "context"],
#             partial_variables={"format_instructions": parser.get_format_instructions()}
#         )
#         print(f"---PROMPT--- {prompt}")
#         # Chain
#         self.rag_chain = prompt | self.llm | parser
    
#     def retrieve(self, state):
#         """
#         Retrieve documents

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): New key added to state, documents, that contains retrieved documents
#         """
#         print("---RETRIEVE---")
#         question = state["question"]

#         # Retrieval
#         documents = self.retriever.get_relevant_documents(question)
#         return {"documents": documents, "question": question}
    
#     def generate(self,state):
#         """
#         Generate response to the user query

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): New key added to state, generation, that contains LLM generation
#         """
#         print("---GENERATE---")
#         question = state["question"]
#         documents = state["documents"]

#         # RAG generation
#         generation = self.rag_chain.invoke({"context": documents, "question": question})
#         print("generation form generate method ", generation)
#         return {"documents": documents, "question": question, "generation": generation}
    
#     def grade_documents(self,state):
#         """
#         Determines whether the retrieved documents are relevant to the question.

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): Updates documents key with only filtered relevant documents
#         """

#         print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#         question = state["question"]
#         documents = state["documents"]

#         # Score each doc
#         filtered_docs = []
#         web_search = "No"
#         for d in documents:
#             score = self.retrieval_grader.invoke(
#                 {"question": question, "document": d.page_content}
#             )
#             grade = score.binary_score
#             if grade == "yes":
#                 print("---GRADE: DOCUMENT RELEVANT---")
#                 filtered_docs.append(d)
#             else:
#                 print("---GRADE: DOCUMENT NOT RELEVANT---")
#                 web_search = "Yes"
#                 continue
#         return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
#     def transform_query(self, state):
#         """
#         Transform the query to produce a better question.

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): Updates question key with a re-phrased questions
#         """

#         print("---TRANSFORM QUERY---")
#         question = state["question"]
#         documents = state["documents"]

#         # Re-write question
#         better_question = self.question_rewriter.invoke({"question": question})
#         return {"documents": documents, "question": better_question}
    
#     def web_search(self,state):
#         """
#         Web search based on the re-phrased question.

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): Updates documents key with appended web results
#         """

#         print("---WEB SEARCH---")
#         question = state["question"]
#         print("inside websearch questions- ",question)
#         documents = state["documents"]
#         # print(documents)

#         # Web search
#         docs = self.google_search.run(question)
#         print(docs)
#         # web_results = "\n".join([d["content"] for d in docs])
#         web_results = Document(page_content=docs)
#         documents.append(web_results)

#         return {"documents": documents, "question": question}
    
#     def decide_to_generate(self,state):
#         """
#         Determines whether to generate an answer, or re-generate a question.

#         Args:
#             state (dict): The current graph state

#         Returns:
#             str: Binary decision for next node to call
#         """

#         print("---ASSESS GRADED DOCUMENTS---")
#         state["question"]
#         web_search = state["web_search"]
#         state["documents"]

#         if web_search == "Yes":
#             # All documents have been filtered check_relevance
#             # We will re-generate a new query
#             print(
#                 "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
#             )
#             return "transform_query"
#         else:
#             # We have relevant documents, so generate answer
#             print("---DECISION: GENERATE---")
#             return "generate"
    
                
#     def __call__(self):
#         self.create_vectors()
#         self.create_grader()
#         self.create_rewriter()
#         self.create_rag_chain()

#         workflow = StateGraph(GraphState)

#         # Define the nodes
#         workflow.add_node("retrieve", self.retrieve)  # retrieve
#         workflow.add_node("grade_documents", self.grade_documents)  # grade documents
#         workflow.add_node("generate", self.generate)  # generatae
#         workflow.add_node("transform_query", self.transform_query)  # transform_query
#         workflow.add_node("web_search_node", self.web_search)  # web search

#         # Build graph
#         workflow.add_edge(START, "retrieve")
#         workflow.add_edge("retrieve", "grade_documents")
#         workflow.add_conditional_edges(
#             "grade_documents",
#             self.decide_to_generate,
#             {
#                 "transform_query": "transform_query",
#                 "generate": "generate",
#             },
#         )
#         workflow.add_edge("transform_query", "web_search_node")
#         workflow.add_edge("web_search_node", "generate")
#         workflow.add_edge("generate", END)


#         self.app = workflow.compile(checkpointer=self.memory)
#         return self.app
        
# if __name__=="__main__":
#     mybot=chatbot()
#     workflow=mybot()
#     config = {"configurable": {"thread_id": "1"}}
#     response=workflow.invoke({"question": "who is prime miniser of USA?"}, config=config)
#     # print(response)
#     print(response["generation"].Response)


# importing a necessary library
from langgraph.graph import StateGraph,MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, TypedDict
from typing import List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.schema import Document
import os
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()

# Access environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"]= GROQ_API_KEY
os.environ["SERPER_API_KEY"]= SERPER_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
class RewriteOutput(BaseModel):
    """Question rewriter to provier improved questions"""
    questions: list[str] = Field(description="Only return List of two rewritten questions")

class FinalOutput(BaseModel):
    """Answer generation to get answer to given question"""
    Response: str = Field(description="Only return Final response from LLM")

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    tool_call: str

class chatbot:
    def __init__(self):
        self.llm=ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
        # self.llm=ChatGroq(model_name="Gemma2-9b-It")
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.google_search = GoogleSerperAPIWrapper()
        self.memory = MemorySaver()
        self.urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            ]   
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=25
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=self.embeddings,
        )
        self.retriever = vectorstore.as_retriever()
        self.retriever_tool=create_retriever_tool(
        self.retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.You are a specialized assistant. Use the 'retriever_tool' **only** when the query explicitly relates to LangChain blog data. For all other queries, respond directly without using any tool. For simple queries like 'hi', 'hello', or 'how are you', provide a normal response.",
        )
        tools=[self.retriever_tool]
        self.retrieve=ToolNode([self.retriever_tool])
        self.llm_with_tool = self.llm.bind_tools(tools)
    def create_grader(self):
        # Prompt
        system1 = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system1),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        self.retrieval_grader = grade_prompt | self.structured_llm_grader
        return self.retrieval_grader
    
    def create_rewriter(self):
        parser = PydanticOutputParser(pydantic_object=RewriteOutput)
        # template = """
        #     You are a question re-writer that improves a user question for web search.
        #     Your task is to generate exactly **two improved versions** of the user's question.
        # template = """Please give me enhanced question for the given input question
        #     **DO NOT** return any reasoning, commentary, or additional information.
        #     Only return the two enhanced questions in the format specified.
        #     Format Instructions:{format_instructions}

        #     Here are some examples:

        #     Original Question: what is AI?
        #     your output: 
        #     What does artificial intelligence mean and how does it work?
        #     What are the basics and key applications of artificial intelligence?

        #     Original Question: how to train model?
        #     your output:
        #     What are the steps to train a machine learning model?
        #     How can I efficiently train an ML model using Python?

        #     Original Question: {question}
        #     your output:
        #     """
        template="Rewrite the input question with slight changes and improved meaning\nQuestion:{question}\nFormat:{format_instructions}"
        prompt = PromptTemplate(
        template = template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        # re_write_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system2),
        #         (
        #             "human",
        #             "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        #         ),
        #     ]
        # )

        self.question_rewriter = prompt | self.llm | parser
        # self.question_rewriter = " ".join(self.question_rewriter)
        return self.question_rewriter
    
    def create_rag_chain(self):
        # # Prompt
        # # prompt = hub.pull("rlm/rag-pro÷mpt")
        # prompt = """You are an assistant for question-answering tasks. 
        #         Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        #         Use five sentences maximum and keep the answer concise.\n
        #         Question: {question} \nContext: {context} \nAnswer:"""

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        parser = PydanticOutputParser(pydantic_object=FinalOutput)
        template = """
            You are an assistant for question-answering tasks.
            Use the following context to answer the question.
            Keep the answer concise and under 8-9 sentences.
            **DO NOT** return any reasoning, commentary, or additional information.
            STRICTLY FORMAT YOUR RESPONSE AS GIVEN BELOW:
            {format_instructions}

            Here are some examples:

            Question: What is LangChain used for?
            Context: LangChain is an open-source framework for building applications with LLMs. It allows developers to connect language models to data sources and tools, making it ideal for tasks like question answering, chatbots, and document analysis.
            Answer:
            1. LangChain is used for building applications with LLMs.
            2. It connects language models to tools and data.
            3. It supports tasks like QA, chatbots, and doc analysis.

            Question: How does vector search work in retrieval systems?
            Context: Vector search uses dense embeddings of text data to find semantically similar content. It relies on nearest neighbor search algorithms over vector representations to retrieve the most relevant documents.
            Answer:
            1. Vector search finds content using semantic similarity.
            2. It uses dense embeddings of text and nearest neighbor algorithms.
            3. Relevant documents are retrieved based on vector closeness.

            Question: {question}
            Context: {context}
            Answer:
            """

        prompt = PromptTemplate(
                template=template,
                input_variables=["question", "context"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
        print(f"---PROMPT--- {prompt}")
        # Chain
        self.rag_chain = prompt | self.llm | parser


    # def ai_assistant(self,state):
    #     print("---CALL AGENT---")
    #     question = state["question"]
    #     response = self.llm_with_tool.invoke(question)
    #     print("ai agent response ", response)

    #     #response=handle_query(messages)
    #     return {"tool_call": "yes", "question":question }
    def ai_assistant(self, state):
        print("---CALL AGENT---")
        parser = PydanticOutputParser(pydantic_object=FinalOutput)
        question = state["question"]

        prompt = PromptTemplate(
            template="""You are a helpful assistant. Answer the user's question.

        Respond ONLY with a JSON object in the following format:
        {format_instructions}

        Question: {question}""",
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        agent_chain = prompt | self.llm_with_tool | parser
        # response = self.llm_with_tool.invoke(question)
        response = agent_chain.invoke(question)
        print("ai agent response ", response)
        
        # Check if response contains indication of tools usage
        # Adjust the condition based on actual response structure/content
        if isinstance(response, str) and "tools" in response.lower():
            tool_call_flag = "yes"
            return {"tool_call": tool_call_flag, "question": question}
        else:
            tool_call_flag = "no"
            return {"tool_call": tool_call_flag, "question": question, "generation":response}

    
    def retriever_func(self, state): 
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        print("in retriver ", question)

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}
    
    def generate(self,state):
        """
        Generate response to the user query

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        print("generation form generate method ", generation)
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased questions
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}
    
    def web_search(self,state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        print("inside websearch questions- ",question)
        documents = state["documents"]
        # print(documents)

        # Web search
        docs = self.google_search.run(question)
        print(docs)
        # web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=docs)
        documents.append(web_results)

        return {"documents": documents, "question": question}
    def custom_tools_condition(self, state):
        print("in custom_tools_condition")
        tool_call = state.get("tool_call")
        print("tool_call ", tool_call)
        if tool_call == "yes":
            return "tools"
        else:
            return END
    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
    
                
    def __call__(self):
        self.create_grader()
        self.create_rewriter()
        self.create_rag_chain()

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retriever_func)  # retrieve
        workflow.add_node("My_Ai_Assistant",self.ai_assistant)
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search_node", self.web_search)  # web search

        # Build graph
        workflow.add_edge(START,"My_Ai_Assistant")
        workflow.add_conditional_edges("My_Ai_Assistant",
                            self.custom_tools_condition,
                            {"tools": "retrieve",
                                END: END})
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)
        

        self.app = workflow.compile(checkpointer=self.memory)

        # img_data = self.app.get_graph(xray=True).draw_mermaid_png()

        # # Save to a file
        # with open("workflow.png", "wb") as f:
        #     f.write(img_data)

        # print("Workflow graph saved as workflow.png")

        return self.app
        
if __name__=="__main__":
    mybot=chatbot()
    workflow=mybot()
    
    config = {"configurable": {"thread_id": "1"}}
    response=workflow.invoke({"question": "Hi my name is Rohan"}, config=config)
    # print(response)
    print(response["generation"].Response)
    