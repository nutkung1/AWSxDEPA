# 🤖 DEPA x AWS

Create and implement a sophisticated chatbot for ExxonMobil that is capable of answering customer inquiries and providing support. The goal is to increase ExxonMobil's revenue in Thailand by promoting and selling lubricant products, even in areas without gas stations.

## :computer: Demo

- Canva: 

## :rocket: Used By

This project is used by the following companies:

- AWS
- DEPA
- J Ventures

## :key: Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`LINE_TOKEN` = Line Message API Token

## :sparkles: Features

Model (LLAMA3-70b)

- Multi-Agent Architecture (8 agents)
- Online Web search
- Speech to Text (Input)
- Text to Speech (Output)
- Car Series Recognition + Prompt to ask the best lubricant for the car at the same time
- Inputting a location to find nearby service centers.
- Searching for service centers.

## :bookmark_tabs:Run Locally

Clone the project

```bash
  git clone https://github.com/nutkung1/AWSxDEPA.git
```

Go to the project directory

```bash
  cd AWSxDEPA
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python run main.py
```

```bash
  ngrok http 10000
```

## :globe_with_meridians:Tech Stack

**Client:** Streamlit

**Tools:** Langchain, OpenAI-Whisper, LLAMA3-70b, GROQ, CarNET

## :envelope_with_arrow:Feedback

If you have any feedback, please reach out to us at suchanat.rata@gmail.com
