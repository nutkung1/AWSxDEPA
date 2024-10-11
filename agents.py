from crewai import Agent, LLM
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
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_text(model_id, body):
    """
    Generate text using your provisioned custom model.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        response (json): The response from the model.
    """

    logger.info("Generating text with your provisioned custom model %s", model_id)

    brt = boto3.client(service_name="bedrock-runtime")

    accept = "application/json"
    content_type = "application/json"

    response = brt.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    logger.info(
        "Successfully generated text with provisioned custom model %s", model_id
    )

    return response_body


def finetune(question):
    """
    Entrypoint for example.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_id = "arn:aws:bedrock:us-east-1:597088053922:provisioned-model/nf4cb3xu0c3v"

    body = json.dumps({"inputText": question})

    response_body = generate_text(model_id, body)
    return response_body["results"][0]["outputText"]


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
    # llm = ChatBedrock(
    #     # model_id="amazon.titan-text-lite-v1",
    #     model_id="arn:aws:bedrock:us-east-1:597088053922:provisioned-model/8az1ox5ovoue",
    #     model_kwargs=dict(temperature=0),
    # )

    # Use a message to the LLM to filter for content violations
    # messages = [
    #     (
    #         "system",
    #         "Please rewrite the input in a way that aligns with AWS Responsible AI Policy and removes any potentially harmful content.",
    #     ),
    #     ("human", text),
    # ]

    # response = llm.invoke(messages)
    # cleaned_text = response.content
    cleaned_text = finetune(text)
    return cleaned_text


def generate_image(text: str) -> str:
    """Generates a business overview image using Amazon Titan and uploads it to S3."""
    llm = ChatBedrock(
        model_id="amazon.titan-text-express-v1",
        model_kwargs=dict(temperature=0),  # Set higher temperature for creativity
    )
    import base64

    text = clean_prompt(text)

    messages = [
        (
            "system",
            "Generate prompt for diagram of the solution for image generation in 40 words.",
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
                "text": concludeText + "In English words",
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "premium",
                "height": 512,
                "width": 512,
                "cfgScale": 3,
                # "seed": 42,
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


# print(generate_image("What is Double whopper"))

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
    kbId = "ULFPGHXRLJ"
    query = question
    answer = bedrock_agent_runtime.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=kbId,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                "overrideSearchType": "HYBRID",  # optional
            }
        },
    )

    # Return the generated answer text
    return answer["retrievalResults"]


from litellm import completion


class ResearchCrewAgents:

    def __init__(self):
        bedrock_client = boto3.client(
            service_name="bedrock-runtime", region_name="us-east-1"
        )

        # Create LLM instance for CrewAI
        self.selected_llm = LLM(
            model="bedrock/meta.llama3-70b-instruct-v1:0",
            custom_llm_provider="bedrock",
            aws_bedrock_client=bedrock_client,
            # temperature=0,
            # max_tokens=280,
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
            max_iter=5,
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
