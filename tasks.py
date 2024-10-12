from crewai import Task


class ResearchCrewTasks:

    def research_task(self, agent, inputs):
        return Task(
            description=(
                f"Based on the {inputs['question']} question, extract information for the question {inputs['question']} "
                "with the help of the tools. Use the Search tool to retrieve information."
                "Verify the accuracy of the information and provide a clear, concise answer."
            ),
            expected_output=(
                f"A clear, concise, and factually accurate answer to the user's question {inputs['question']},"
                "based on the information retrieved from the tool use. Don't make up an answer."
                f"If the {inputs['question']} question not related information retrieved from the tool just say 'The answer can be incorrect or not related to the question', but don't be too strict, if it not 100% related, 80% is acceptable."
            ),
            agent=agent,
            async_execution=True,
        )

    def writing_task(self, agent, context, inputs):
        return Task(
            description=(
                "Use the verified research information provided by the Research Agent."
                "Your task is to create a well-structured and engaging piece of content."
                "Focus on clarity, readability, and flow. The content should be suitable for"
                "the intended audience and the topic should be covered comprehensively."
                "Ensure that the final content is formatted and ready for publication and use all the key points to write the answer."
            ),
            expected_output=(
                "A complete and engaging piece of content, maximum 4 sentences."
                "that is well-structured, easy to read, and aligns with the information provided. The answer you write need to describe How to (each step process) of the method and solution appraoch. Don't imply. Write the answer in a friendly and engaging tone."
                f"If the {inputs['question']} question not related information retrieved from the research agents just say 'Unfortunately, I could not find any relevant information on this topic'."
                f"The answer need to use the context {context} and write the best friendly answer related to the question {inputs['question']}"
            ),
            agent=agent,
            context=context,
            # async_execution=True,
        )

    def hallucination_task(self, agent, context, inputs):
        return Task(
            description=(
                f"Based on the response from the grader task for the quetion {inputs['question']} evaluate whether the answer is grounded in / supported by a set of facts."
            ),
            expected_output=(
                # "Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
                "if the answer is in useful and contains fact about the question asked just give the answer of content writing task."
                "Respond 'Sorry, I cannot find relevant information from the database.' if the answer is not useful and does not contains fact about the question asked."
                # "Do not provide any preamble or explanations except for 'yes' or 'no'."
            ),
            agent=agent,
            context=context,
            async_execution=True,
        )
