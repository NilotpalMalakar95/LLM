from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from agents.wikipedia_lookup_agent import lookup as wikipedia_lookup_agent
from output_parsers import data_parser


def get_wikipedia_data(keyword: str) -> str:
    load_dotenv()

    # Fetching information
    wikipedia_content = wikipedia_lookup_agent(keyword=keyword)

    # keywords inside curly-braces denotes the programmable parameters in a prompt
    # We can think of it like an f-string
    summary_template = """given a keyword {keyword} 
    I want you to
    1. Create a short summary on the keyword
    2. Give 3 most important factors to be known about the keyword
    Note: Make sure to keep the word limit within 300 words and keep the answers as short and as technical as possible
    \n{format_instructions}"""

    # The Prompt Template should contain at-least these two things
    # 1. input_variables = list of the programmable variables that we are going to use in the summary template for the prompt; should be in string format; should be exactly same as in the summary template
    # 2. Summary template = the summary template with programmable variables that would form the prompt; Text before we inject it with variables
    summary_prompt_template = PromptTemplate(
        input_variables=["keyword"],
        template=summary_template,
        partial_variables={
            "format_instructions": data_parser.get_format_instructions()
        },
    )

    # Instantiating the LLM object now
    # Arguments:-
    # model_name : name of the model to be used
    # temperature : decides how creative the model can be : 0 --> No creativity, 1 --> highest creativity
    llm_model = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

    # Instantiate the chain
    # Arguments:-
    # 1. llm : model variable : LLM wrapped inside the ChatModel abstraction
    # 2. prompt : engineered prompt
    chain = LLMChain(llm=llm_model, prompt=summary_prompt_template)

    # Feeding information to the prompt template
    # Here we would have to feed the data for the programmable variables in the prompt, this would be fed to the LLM and the LLM would later produce results as per the data been fed
    result = chain.invoke(input={"keyword": wikipedia_content})

    # # Print the results
    # print(result.get("text"))
    # print(result)
    return result.get("text")


if __name__ == "__main__":
    result = get_wikipedia_data(keyword="Adolf Hitler")
    print(result)
