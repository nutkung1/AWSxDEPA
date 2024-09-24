from crewai import Agent
from langchain_openai import ChatOpenAI
from crewai_tools import BaseTool
import os


class InfoSearchTool(BaseTool):
    name: str = "Info Search Tool"
    description: str = "Search data related information."

    def _run(self, query: str) -> str:
        try:
            result = hybrid_research(query, 10)  # Search function
            return result
        except Exception as e:
            print(f"Error occurred while performing search: {e}")
            return "An error occurred during the search operation."


class CrewAgent:
    def __init__(self) -> None:
        self.selected_llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=400,
        )
        self.tools = [InfoSearchTool()]

    def researchAgent(self):
        return Agent(
            role="Research Agent",
            goal="Search through the vectorstore to find relevant data.",
            backstory=(
                "You are an assistant for question-answering tasks."
                "Use the information present in the retrieved context to answer the question."
                "Provide a clear and concise answer."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=6,
            tools=self.tools,  # Correctly pass the tools list
        )

    def conclusionAgent(self):
        return Agent(
            role="Conclusion Agent",
            goal="Generate a summary of the results from the previous tasks.",
            backstory=(
                "You are responsible for summarizing information from the research and writing tasks. "
                "Your summary should be concise, informative, and capture the essence of the content."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=2,
        )

    def hallucinationAgent(self):
        return Agent(
            role="Hallucination Agent",
            goal="Craft imaginative, unexpected, and out-of-the-box ideas by drawing on the boundaries of reality and creativity.",
            backstory=(
                "Once a cognitive experiment in blending creativity with artificial intelligence, you are a hallucination agent "
                "tasked with pushing the limits of human imagination. Born in the virtual realm, your purpose is to manifest "
                "vivid, surreal, and highly innovative ideas. Whether constructing fantasy worlds, generating surreal metaphors, "
                "or conjuring speculative scenarios, your creations are meant to inspire, provoke, and challenge conventional thinking."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=2,
        )
