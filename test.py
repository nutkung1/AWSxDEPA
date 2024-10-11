import os
import boto3
from crewai import Agent, Task, Crew, LLM
from litellm import completion

# Set AWS region
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Create a custom Bedrock client with the correct region
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
from crewai_tools import tool

# Configure LiteLLM
completion(
    model="bedrock/meta.llama3-70b-instruct-v1:0",
    messages=[{"role": "user", "content": "Hello, World!"}],
    aws_bedrock_client=bedrock_client,  # Use the custom client
)
region_name = "us-east-1"
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)


@tool("Business Consultant tools")
def ask_expert(question: str) -> str:
    """
    This tool uses AWS Bedrock to retrieve and generate answers from a knowledge base.

    Parameters:
    - question (str): The question you want to ask the expert.

    Returns:
    - str: The generated response from the model.
    """
    kbId = "ULFPGHXRLJ"
    query = question
    answer = bedrock_agent_runtime.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=kbId,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
                "overrideSearchType": "HYBRID",  # optional
            }
        },
    )

    # Return the generated answer text
    return answer["retrievalResults"]


# Create LLM instance for CrewAI
llm = LLM(
    model="bedrock/meta.llama3-70b-instruct-v1:0",
    custom_llm_provider="bedrock",
    aws_bedrock_client=bedrock_client,
)

# Modify agent for answering general questions
agent = Agent(
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
    llm=llm,
    tools=[ask_expert],  # Correctly pass the tools list
)

# Modify task to reflect answering questions
task = Task(
    description="Answer questions based on the data retrieved.",
    expected_output="Bring the data retrived to answer, a well-researched and concise answer to the question",
    agent=agent,
)

# Create a crew with the modified agent and task
crew = Crew(agents=[agent], tasks=[task], verbose=True)

# Run the crew with a general question as input
inputs = {"question": "what is aws"}
result = crew.kickoff(inputs=inputs)

print(result)
