from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

# Suppress warnings
set_verbosity_error()
class AIAssistant:
    def __init__(self):
        # Initialize summarization
        self.summarization_pipeline = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn"
        )
        self.summarizer = HuggingFacePipeline(pipeline=self.summarization_pipeline)
        
        # Initialize Q&A
        self.qa_pipeline = pipeline(
            "question-answering", 
            model="distilbert-base-uncased-distilled-squad"
        )
        
        # Create summarization chain
        self.summary_template = PromptTemplate.from_template(
            "Summarize the following text in a {length} way:\n\n{text}"
        )
        self.summarization_chain = self.summary_template | self.summarizer
    
    def summarize(self, text, length="medium"):
        """Summarize text with specified length"""
        try:
            summary = self.summarization_chain.invoke({
                "text": text, 
                "length": length
            })
            return summary.strip()
        except Exception as e:
            return f"Error in summarization: {str(e)}"
    
    def answer_question(self, question, context):
        """Answer questions based on given context"""
        try:
            result = self.qa_pipeline(question=question, context=context)
            return {
                "answer": result["answer"],
                "confidence": round(result["score"], 3)
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "confidence": 0}
def main():
    print("🤖 AI Assistant - Text Summarization & Q&A")
    print("=" * 50)
    
    assistant = AIAssistant()
    
    # Get text to summarize
    print("\n📝 Enter text to summarize:")
    text = input()
    
    # Get summary length preference
    print("\n📏 Choose summary length (short/medium/long):")
    length = input().strip().lower()
    if length not in ["short", "medium", "long"]:
        length = "medium"
    
    # Generate summary
    print(f"\n🔄 Generating {length} summary...")
    summary = assistant.summarize(text, length)
    
    print(f"\n✨ Summary:")
    print("-" * 30)
    print(summary)
    print("-" * 30)
    
    # Interactive Q&A
    print(f"\n❓ Ask questions about the summary (type 'exit' to quit):")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        # Get answer
        result = assistant.answer_question(question, summary)
        
        print(f"\n💡 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']}")
if __name__ == "__main__":
    main()