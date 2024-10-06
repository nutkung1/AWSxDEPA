from crewai import Agent
import os
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
import json

load_dotenv(override=True)
# Initialize the search tool with the specified directory and model configuration
from crewai_tools import tool

import boto3

region_name = "us-east-1"
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region_name)


from langchain_aws import ChatBedrock


def upload_to_s3(image_data: bytes, filename: str) -> str:
    """Uploads the image directly to S3 from memory and returns the S3 URL."""
    s3 = boto3.client("s3")
    bucket_name = "depaimagegen"  # Replace with your S3 bucket name

    # Upload image bytes directly to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=filename,
        Body=image_data,
        ContentType="image/png",  # Change to appropriate MIME type
        ACL="public-read",  # Make the object publicly accessible
    )

    # Construct the public S3 URL for the uploaded image
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"

    return s3_url


def clean_prompt(text: str) -> str:
    """Cleans input text to avoid content filter violations."""
    llm = ChatBedrock(
        model_id="amazon.titan-text-lite-v1",
        model_kwargs=dict(temperature=0),
    )

    # Use a message to the LLM to filter for content violations
    messages = [
        (
            "system",
            "Please rewrite the input in a way that aligns with AWS Responsible AI Policy and removes any potentially harmful content.",
        ),
        ("human", text),
    ]

    response = llm.invoke(messages)
    cleaned_text = response.content
    return cleaned_text


def generate_image(text: str) -> str:
    """Generates a business overview image using Amazon Titan and uploads it to S3."""
    llm = ChatBedrock(
        model_id="amazon.titan-text-lite-v1",
        model_kwargs=dict(temperature=0),  # Set higher temperature for creativity
    )
    import base64

    text = clean_prompt(text)

    messages = [
        (
            "system",
            "Summarize the business overview response with key elements like industry, goals, products, customers, and challenges for image generation in 50 words.",
        ),
        ("human", text),
    ]
    ai_msg = llm.invoke(messages)
    concludeText = ai_msg.content
    print("Image Generation Prompt:", concludeText)

    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": concludeText,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "height": 512,
                "width": 512,
                "cfgScale": 5,
                "seed": 42,
            },
        }
    )

    bedrock = boto3.client(service_name="bedrock-runtime")
    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-image-generator-v2:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    base64_image = response_body.get("images")[0]

    # Convert base64 image to bytes for S3 upload
    image_bytes = base64.b64decode(base64_image)

    # Upload the image to S3 and get the URL
    image_url = upload_to_s3(image_bytes, "generated_business_image.png")

    return image_url


# print(generate_image("What is Double Hamburger"))

from langchain_openai import ChatOpenAI

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


@tool("Business Consultant tools")
def ask_expert(question: str) -> str:
    """
    This tool uses AWS Bedrock to retrieve and generate answers from a knowledge base.

    Parameters:
    - question (str): The question you want to ask the expert.

    Returns:
    - str: The generated response from the model.
    """

    model_id = "amazon.titan-text-premier-v1:0"
    model_arn = f"arn:aws:bedrock:{region_name}::foundation-model/{model_id}"
    kbId = "ULFPGHXRLJ"
    query = question
    return bedrock_agent_runtime.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=kbId,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
                "overrideSearchType": "HYBRID",  # optional
            }
        },
    )


class ResearchCrewAgents:

    def __init__(self):
        # SELECT YOUR MODEL HERE
        self.selected_llm = ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama-3.2-90b-text-preview",
            temperature=0,
            max_tokens=300,
        )

        """Embedded"""
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            model_kwargs={
                "dimensions": 1024,
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
            tools=[ask_expert],  # Correctly pass the tools list
        )

    def writer(self):
        # Setup the Writer agent
        return Agent(
            role="Content Writer",
            goal="Write engaging content based on the provided research or information.",
            backstory=(
                "You are a professional in Business overview writer. You need to write the content in a way that is engaging and informative for Business customer."
                "Also, you are a skilled writer who excels at turning raw data into captivating narratives."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.selected_llm,
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
