#Install the langchain, OpenAI and google-search-results packages:
pip install langchain
pip install openai
pip install google-search-results

#Import the os module and set the OpenAI API key as an environment variable:
import os
os.environ["OPENAI_API_KEY"] = "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"

#Import necessary modules from langchain:
from langchain.llms import openAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent
from langchain import ConversationChain

#Create an instance of the openAI LLM model:
llm = openAI(temperature=0.7)

#Specify a text prompt and generate output using the LLM model:
text = "What are 5 vacation destinations for someone who likes to eat pasta?"
print(llm(text))

#Create a prompt template for dynamic inputs:
prompt = PromptTemplate(
    input_variables=["food"],
    template="What are 5 vacation destinations for someone who likes to eat {food}?",
)

#Generate output using the prompt template and LLM model:
print(prompt.format(food="sambar rice"))
print(llm(prompt.format(food="sambar rice")))

#Create an instance of the LLMChain class:
chain = LLMChain(llm=llm, prompt=prompt)

#Run the chain with a specific input:
print(chain.run("mangos"))


#Load necessary tools and initialize an agent with the OpenAI LLM model:
os.environ["SERPAPI_API_KEY"] = "162943501ff761253508e7932c00f271"
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

#Run the agent with specific queries:
agent.run("Who is the current leader of Japan? What is the largest prime number that is smaller than their age?")

#Create an instance of the ConversationChain class:
conversation = ConversationChain(llm=llm, verbose=True)

#Interact with the conversation chain:
conversation.predict(input="Hi there!")
# conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# conversation.predict(input="What was the first thing I said to you?")
# conversation.predict(input="What was an alternative phrase for the first thing I said to you?")
# conversation.predict(input="Hi there! How are you?")
# conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# conversation.predict(input="What was the first thing I said to you?")
# conversation.predict(input="What was the first thing I said to you?")
# conversation.predict(input="What was the thing I said to you recently?")

