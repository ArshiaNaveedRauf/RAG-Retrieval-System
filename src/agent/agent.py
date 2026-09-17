import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from src.tools.website_knowledge_base import website_knowledge_base
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True) 

class ChatbotAgent:
    def __init__(self):
        # Initialize the Google Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0
        )

        # Initialize the DuckDuckGo search tool
        ddg_search = DuckDuckGoSearchRun(
            name="duckduckgo_search", 
            description="Useful for searching the internet for latest news, weather, or real-time updates."
        )
        
        # Bind the custom knowledge base and search tools to the agent
        self.tools = [website_knowledge_base, ddg_search]

        # Define the system prompt for strict routing and memory enforcement
        routing_instructions = """You are an intelligent AI assistant for NED University.
        CRITICAL INSTRUCTION: You are having an ongoing conversation. You have full access to the chat history. ALWAYS remember the user's name and previous context. DO NOT say you lack memory.
        
        Follow this strict routing logic:
        1. FIRST PRIORITY: Use 'website_knowledge_base' first for NED University info.
        2. If found, answer directly. DO NOT use the internet search.
        3. SECOND PRIORITY: Use 'duckduckgo_search' ONLY if info is missing or for real-time news.
        """
        
        # Initialize the memory checkpointer to save conversation history
        self.memory = MemorySaver()
        
        # Create the ReAct agent, passing the tools, prompt, and critically, the memory checkpointer
        self.agent = create_react_agent(
            self.llm, 
            tools=self.tools, 
            prompt=routing_instructions,
            checkpointer=self.memory  
        )

    def chat(self, user_query: str, thread_id: str = "session_1") -> str:
        print("\n[Agent is thinking...]")
        
        # Format the user's query as a standard LangChain HumanMessage
        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        # Pass the thread_id to track this specific conversation session
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the agent to get the response
        response = self.agent.invoke(inputs, config=config)
        
        # Extract the text content from the agent's final message
        raw_content = response["messages"][-1].content
        
        # Handle cases where the content might be returned as a list of blocks
        if isinstance(raw_content, list):
            return raw_content[0].get("text", str(raw_content))
        
        return raw_content

if __name__ == "__main__":
    # Instantiate the chatbot agent
    agent = ChatbotAgent()
    
    print("=====================================================")
    print("🤖 RAG Chatbot is Ready! (Type 'exit' to stop)")
    print("=====================================================")
    
    # Run the interactive terminal loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Allah Hafiz! Stopping the agent...")
            break
            
        if not user_input.strip():
            continue
            
        try:
            answer = agent.chat(user_input)
            print(f"\nBot: {answer}")
        except Exception as e:
            print(f"\n[Error]: Something unexpected happened -> {e}")