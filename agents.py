from crewai import Agent
import os
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings

load_dotenv(override=True)
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


from langchain_openai import ChatOpenAI

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class ResearchCrewAgents:

    def __init__(self):
        # SELECT YOUR MODEL HERE
        self.selected_llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama-3.2-90b-text-preview",
            temperature=0,
            max_tokens=280,
        )
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            model_kwargs={
                "dimensions": 1024,
                "normalize": True,
            },
        )

    def researcher(self):
        # Setup the tool for the Researcher agent
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
            tools=[ask_expert],  # Correctly pass the tools list
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
