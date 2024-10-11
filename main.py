from pydantic import BaseModel
import os
from crewai import Crew, Process
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks
from dotenv import load_dotenv


load_dotenv(override=True)


# Setup environment variables
def setup_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PORT"] = os.getenv("PORT", "10000")


setup_environment()


class ResearchCrew:
    def __init__(self, inputs):
        self.inputs = inputs
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

    def serialize_crew_output(self, crew_output):
        return {"output": crew_output}

    def run(self):

        researcher = self.agents.researcher()
        writer = self.agents.writer()
        hallucinator = self.agents.hallucination()

        research_task = self.tasks.research_task(researcher, self.inputs)
        writing_task = self.tasks.writing_task(writer, [research_task], self.inputs)
        hallucination_task = self.tasks.hallucination_task(
            hallucinator, [writing_task], self.inputs
        )

        crew = Crew(
            agents=[researcher, writer, hallucinator],
            tasks=[research_task, writing_task, hallucination_task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "aws_bedrock",
                "config": {
                    "model": "amazon.titan-embed-text-v2:0",
                    "vector_dimension": 1024,
                },
            },
        )
        self.result = crew.kickoff(inputs=self.inputs)

        self.serialized_result = self.serialize_crew_output(self.result)
        return {"result": self.serialized_result}


class QuestionRequest(BaseModel):
    question: str


USELESS_INFO_PHRASES = [
    "I don't know",
    "does not contain information",
    "does not contain any information",
    "any information",
    "Unfortunately",
    "Agent stopped due to iteration limit or time limit",
]


def has_useful_information(result):
    return "Agent stopped due to iteration limit or time limit" not in result


from flask import Flask, request
import os
import json
import requests

# ------------ end import zone -----------

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


# ------------insert new code below --------


@app.route("/webhook", methods=["POST"])
# for checking that the request
def webhook():
    req = request.json
    print("request coming from line")
    print(req)
    if len(req["events"]) == 0:
        return "", 200

    handleRequest(req)
    return "", 200


from agents import generate_image


def handleRequest(req):
    token = os.environ.get("LINE_TOKEN")  # edit token here
    reply_url = "https://api.line.me/v2/bot/message/reply"
    Authorization = "Bearer {}".format(token)

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": Authorization,
    }

    response = handleEvents(req["events"][0], req["destination"])
    replyToken = req["events"][0]["replyToken"]

    if "cannot find relevant information" in response:
        data = json.dumps(
            {
                "replyToken": replyToken,
                "messages": [
                    {
                        "type": "text",
                        "text": response,
                    }
                ],
            }
        )
        r = requests.post(reply_url, headers=headers, data=data)
        print(f"Response for no information: {r.text}")  # Log response
    else:
        image_url = generate_image(str(response))

        print(f"Generated image URL: {image_url}")  # Debug: check generated URL

        data = json.dumps(
            {
                "replyToken": replyToken,
                "messages": [
                    {
                        "type": "text",
                        "text": response,
                    },
                    {
                        "type": "image",
                        "originalContentUrl": image_url,
                        "previewImageUrl": image_url,
                    },
                ],
            }
        )

        r = requests.post(reply_url, headers=headers, data=data)
        print(f"LINE API Response: {r.text}")  # Debug: check LINE API response


chat_conver = {}


def handleEvents(event, destination):
    if destination not in chat_conver:
        chat_conver[destination] = []

    if event["message"]["type"] == "text":
        return handleMessage(event["message"], destination)
    else:
        print(f"Unknown event type: {event['type']}")
        return "sorry unknown type format we still can't handle this type of message"


from langdetect import detect
from deep_translator import GoogleTranslator


def handleMessage(event, destination):

    all_language = {
        "en": "en",
        "th": "th",
    }
    textFromUser = event["text"]
    new_message = {"type": "msg", "msg": textFromUser}
    lang = detect(textFromUser)
    # output = Translator().translate(textFromUser, dest=f'{all_language[lang]}').text

    print("chat convert destination")
    print(chat_conver[destination])
    # translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome")
    if len(chat_conver[destination]) == 0:
        # Initialize chat_conver[destination] if it doesn't exist
        chat_conver[destination] = []
        chat_conver[destination].append(new_message)
        inputs = textFromUser
        result = ask_question(inputs)
        result = (
            result["result"]["output"]
            .raw.replace("yes.", "")
            .replace("yes", "")
            .replace("Yes.", "")
            .strip()
        )
        print("result returning from latest message text", result)
        if lang == "th":
            return GoogleTranslator(source="auto", target="th").translate(result[""])
        else:
            return result
    else:
        chat_conver[destination].append(new_message)
        inputs = textFromUser
        result = ask_question(inputs)
        result = (
            result["result"]["output"]
            .raw.replace("yes.", "")
            .replace("yes", "")
            .replace("Yes.", "")
            .strip()
        )
        print("result returning from latest message text", result)
        if lang == "th":
            return GoogleTranslator(source="auto", target="th").translate(result)
        else:
            return result


def ask_question(question):
    try:
        research_crew = ResearchCrew({"question": question})
        result = research_crew.run()
        return result
    except Exception as e:
        print(f"Error during question handling: {e}")
        return {"result": {"output": "An error occurred during processing."}}


# ------------ end edit zone  --------
""" gunicorn -w 4 -b 127.0.0.1:10000 main:app """
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", "10000")))
