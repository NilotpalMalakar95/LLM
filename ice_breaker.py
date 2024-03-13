import os
from dotenv import load_dotenv

from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv()

    information = """Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect, and former chairman of Tesla, Inc.; owner, executive chairman, and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is one of the wealthiest people in the world, with an estimated net worth of US$190 billion as of March 2024, according to the Bloomberg Billionaires Index, and $195 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]
A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial intelligence company.
Musk has expressed views that have made him a polarizing figure.[7] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and antisemitic conspiracy theories.[7][8][9][10] His ownership of Twitter has been similarly controversial, being marked by the laying off of a large number of employees, an increase in hate speech and misinformation and disinformation on the website, as well as changes to Twitter Blue verification."""

    # keywords inside curly-braces denotes the programmable parameters in a prompt
    # We can think of it like an f-string
    summary_template = """Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    # The Prompt Template should contain at-least these two things
    # 1. input_variables = list of the programmable variables that we are going to use in the summary template for the prompt; should be in string format; should be exactly same as in the summary template
    # 2. Summary template = the summary template with programmable variables that would form the prompt; Text before we inject it with variables
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Instantiating the LLM object now
    # Arguments:-
    # model_name : name of the model to be used
    # temperature : decides how creative the model can be : 0 --> No creativity, 1 --> highest creativity
    llm_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Instantiate the chain
    # Arguments:-
    # 1. llm : model variable : LLM wrapped inside the ChatModel abstraction
    # 2. prompt : engineered prompt
    chain = LLMChain(llm=llm_model, prompt=summary_prompt_template)

    # Feeding information to the prompt template
    # Here we would have to feed the data for the programmable variables in the prompt, this would be fed to the LLM and the LLM would later produce results as per the data been fed
    result = chain.invoke(input={"information": information})
