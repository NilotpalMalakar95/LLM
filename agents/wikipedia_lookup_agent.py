from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor

from tools.tools import get_wikipedia_data


def lookup(keyword: str) -> str:
    llm_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """given a keyword {keyword} 
    I want you to
    1. Create a short summary on the keyword
    2. Give 3 most important factors to be known about the keyword
    Note: Make sure to keep the word limit within 300 words and keep the answers as short and as technical as possible"""

    # template = """given a keyword {keyword} I want you to summarize everything that you can find in a paragraph or 10 bullet points and 300 words maximum.
    # If you are creating bullet points follow the following steps:
    # 1. The points should not be too large and should be technical and professional.\
    # 2. Start each point with a number example 1,2,3..."""

    tools_for_agent = [
        Tool(
            name="crawl wikipedia to gather data",
            func=get_wikipedia_data,
            description="Useful for when you want to get data from wikipedia on any prescribed topics",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm_model, tools=tools_for_agent, prompt=react_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=tools_for_agent, handle_parsing_errors=True, verbose=True
    )

    prompt_template = PromptTemplate(template=template, input_variables=["keyword"])

    result = agent_executor.invoke(
        {"input": prompt_template.format_prompt(keyword=keyword)}
    )

    wikipedia_data = result["output"]

    return wikipedia_data
