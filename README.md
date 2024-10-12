# 🤖 DEPA x AWS

This project is designed to enhance J Ventures' revenue by offering customers access to a business consultant chatbot.

## :computer: Demo

- Canva: https://www.canva.com/design/DAGSJrVTtOk/ew3bP0uA4l7IsF4JXl9Q2w/edit?utm_content=DAGSJrVTtOk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## :rocket: Used By

This project is used by the following companies:

- AWS
- DEPA
- J Ventures

## :key: Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`LINE_TOKEN` = Line Message API Token

## :sparkles: Models

- Model (Amazon Titan Text G1)
- Model (Amazon Titan Image Generator G1 V2)
- Model (LLAMA3-70b)

## ❇️ Features
- Multi-Agent Architecture (5 agents)
- Hybrid Search
- Image generator

## :bookmark_tabs: Run Locally

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

Start the python

```bash
  python run main.py
```
Start the server

```bash
  ngrok http 10000
```

## :globe_with_meridians: Tech Stack

**Client:** Line Official Account

**Tools:** CrewAI, AWS Bedrock, AWS S3

## :envelope_with_arrow:Feedback

If you have any feedback, please reach out to us at suchanat.rata@gmail.com
