from crewai import Agent
import os
from citation import Citation
from dotenv import load_dotenv


load_dotenv(override=True)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Initialize the search tool with the specified directory and model configuration
from crewai_tools import tool


@tool("Business Consultant Expert")
def ask_expert(question: str) -> str:
    """
    This tool uses AWS Bedrock to retrieve and generate answers from a knowledge base using the Business Consultant Expert model.

    Parameters:
    - question (str): The question you want to ask the expert.

    Returns:
    - str: The generated response from the model.
    """
    region_name = "us-east-1"
    import boto3

    bedrock_agent_runtime = boto3.client(
        "bedrock-agent-runtime", region_name=region_name
    )

    model_id = "amazon.titan-text-premier-v1:0"
    model_arn = f"arn:aws:bedrock:{region_name}::foundation-model/{model_id}"
    kbId = "EYWMBRPL3V"
    query = question
    return bedrock_agent_runtime.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=kbId,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                "overrideSearchType": "HYBRID",  # optional
            }
        },
    )


class ResearchCrewAgents:

    def __init__(self):
        # Initialize the LLM to be used by the agents
        self.cite = Citation()
        # SELECT YOUR MODEL HERE
        self.selected_llm = self.cite.llm

    def researcher(self):
        # Setup the tool for the Researcher agent
        tools = [ask_expert]
        return Agent(
            role="Research Agent",
            goal="Search through the data to find relevant and accurate answers.",
            backstory=(
                "You are an assistant for question-answering tasks."
                "Use the information present in the retrieved context to answer the question."
                "Provide a clear and concise answer."
                "Do not remove technical terms that are important for the answer, as this could make it out of context."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=10,
            tools=tools,  # Correctly pass the tools list
        )

    def writer(self):
        # Setup the Writer agent
        return Agent(
            role="Content Writer",
            goal="Write engaging content based on the provided research or information.",
            backstory=(
                "You are a professional in Optimism which is a Collective of companies, communities, and citizens working together to reward public goods and build a sustainable future for Ethereum."
                "Also, you are a skilled writer who excels at turning raw data into captivating narratives."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
            max_iter=1,
        )

    def hallucination(self):
        # Setup the Conclusion agent
        return Agent(
            role="Hallucination Grader",
            goal="Filter out hallucination",
            backstory=(
                "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
                "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
        )
