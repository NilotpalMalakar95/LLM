from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai.chat_models import ChatOpenAI

if __name__ == "__main__":
    load_dotenv()

    model = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

    raw_prompt = """Given a topic : {topic}
    Summarize the topic within 3000 words or as per the requirement"""
    prompt_template = PromptTemplate(input_variables=["topic"], template=raw_prompt)

    chain = LLMChain(llm=model, prompt=prompt_template)

    result = chain.invoke(
        input={
            "topic": "Methods to calculate correlations between categorical and continuous data using python"
        }
    )

    print(result.get("text"))
