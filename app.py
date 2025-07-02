# ADVANCE LANGCHAIN + HUGGINGFACE AND TRANSFORMERS
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()  # Suppress warnings from transformers

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn") # Initialize the summarization pipeline with a specific model
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline) # Wrap the summarization pipeline as a LangChain model

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad") # Question answering pipeline

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summary_template | summarizer

text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n🔹 **Generated Summary:**")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n🔹 **Answer:**")
    print(qa_result["answer"]) # Display the answer from the model