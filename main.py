from transformers import pipeline

# model = pipeline("summarization", model="facebook/bart-large-cnn")
# long_text = """
# The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
# It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
# Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially 
# criticized by some of France's leading artists and intellectuals for its design, but it has 
# become a global cultural icon of France and one of the most recognizable structures in the world. 
# The Eiffel Tower is the most-visited paid monument in the world; 6.91 million people ascended 
# it in 2015. The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey 
# building, and the tallest structure in Paris.
# """

# response = model(long_text,min_length=30,max_length=38) 
# print(response)


# LANGCHAIN

from langchain_huggingface import HuggingFacePipeline
# this allow me to wrap a hugging face model as a langchain model
from langchain.prompts import PromptTemplate

model = pipeline("text-generation", 
                 model = "gpt2", # "gpt2" is a smaller model,
                  # mistral/Mistral-7B-Instruct-v0.1 => larger model
                 device=0, # use GPU if available
                 max_length=256, 
                 truncation=True,
                 )
llm = HuggingFacePipeline(pipeline=model) # wrap the model as a langchain model

template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand.")
chain = template | llm # create a chain with the template and the model
topic = input("Enter a topic: ")
age = input("Enter the age of the user: ")

# Execute the chain
response = chain.invoke({"topic": topic, "age": age})
print(response)