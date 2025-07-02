import sys
from datetime import datetime
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Suppress transformer warnings
set_verbosity_error()
class AITextProcessor:
    """A comprehensive text processing application using HuggingFace and LangChain"""
    
    def __init__(self):
        self.models = {}
        self.setup_models()
    
    def setup_models(self):
        """Initialize all required models"""
        print("🔄 Loading AI models...")
        
        try:
            # Summarization
            self.models['summarizer'] = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn"
            )
            
            # Question Answering
            self.models['qa'] = pipeline(
                "question-answering", 
                model="distilbert-base-uncased-distilled-squad"
            )
            
            # Text Generation
            text_gen_pipeline = pipeline(
                "text-generation", 
                model="gpt2",
                max_length=256,
                truncation=True,
                pad_token_id=50256
            )
            self.models['generator'] = HuggingFacePipeline(pipeline=text_gen_pipeline)
            
            # Create prompt template
            self.explanation_template = PromptTemplate.from_template(
                "Explain {topic} in simple terms for a {age} year old:"
            )
            self.explanation_chain = self.explanation_template | self.models['generator']
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            sys.exit(1)
    
    def summarize_text(self, text, length="medium"):
        """Summarize text with error handling"""
        length_configs = {
            "short": {"max_length": 50, "min_length": 10},
            "medium": {"max_length": 130, "min_length": 30},
            "long": {"max_length": 200, "min_length": 50}
        }
        
        config = length_configs.get(length, length_configs["medium"])
        
        try:
            # Validate input
            if len(text.strip()) < 50:
                return "Error: Text too short for summarization (minimum 50 characters)"
            
            result = self.models['summarizer'](
                text,
                max_length=config["max_length"],
                min_length=config["min_length"],
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            return f"Summarization error: {str(e)}"
    
    def answer_question(self, question, context):
        """Answer questions with confidence scoring"""
        try:
            result = self.models['qa'](question=question, context=context)
            return {
                "answer": result["answer"],
                "confidence": round(result["score"], 3),
                "start": result.get("start", 0),
                "end": result.get("end", 0)
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "start": 0,
                "end": 0
            }
    
    def generate_explanation(self, topic, age):
        """Generate age-appropriate explanations"""
        try:
            response = self.explanation_chain.invoke({
                "topic": topic,
                "age": age
            })
            
            # Clean up the response
            if isinstance(response, str):
                # Extract the explanation part after the prompt
                lines = response.split('\n')
                explanation = []
                for line in lines:
                    if line.strip() and not line.startswith("Explain"):
                        explanation.append(line.strip())
                
                return ' '.join(explanation) if explanation else response
            
            return str(response)
            
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def save_session(self, content, session_type="summary"):
        """Save session results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_type}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"AI Text Processor - {session_type.title()} Session\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)
            
            return filename
        except Exception as e:
            return f"Error saving file: {str(e)}"
def main():
    """Main application interface"""
    print("🤖 AI Text Processor")
    print("Powered by HuggingFace Transformers + LangChain")
    print("=" * 60)
    
    processor = AITextProcessor()
    
    while True:
        print("\n📋 Choose an option:")
        print("1. 📝 Summarize Text")
        print("2. ❓ Q&A Session")
        print("3. 🎓 Generate Explanation")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            handle_summarization(processor)
        elif choice == "2":
            handle_qa_session(processor)
        elif choice == "3":
            handle_explanation(processor)
        elif choice == "4":
            print("👋 Thank you for using AI Text Processor!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
def handle_summarization(processor):
    """Handle text summarization workflow"""
    print("\n📝 TEXT SUMMARIZATION")
    print("-" * 30)
    
    # Get text input
    print("Enter text to summarize (or 'back' to return):")
    text = input().strip()
    
    if text.lower() == 'back':
        return
    
    if not text:
        print("❌ No text provided.")
        return
    
    # Get length preference
    print("\nSummary length? (short/medium/long) [default: medium]:")
    length = input().strip().lower()
    if length not in ["short", "medium", "long"]:
        length = "medium"
    
    # Generate summary
    print(f"\n🔄 Generating {length} summary...")
    summary = processor.summarize_text(text, length)
    
    print(f"\n✨ SUMMARY ({length}):")
    print("=" * 40)
    print(summary)
    print("=" * 40)
    
    # Option to save
    save_choice = input("\n💾 Save summary to file? (y/n): ").strip().lower()
    if save_choice == 'y':
        content = f"Original Text:\n{text}\n\nSummary ({length}):\n{summary}"
        filename = processor.save_session(content, "summary")
        print(f"✅ Saved as: {filename}")
def handle_qa_session(processor):
    """Handle Q&A session workflow"""
    print("\n❓ QUESTION & ANSWER SESSION")
    print("-" * 30)
    
    # Get context
    print("Enter the text/context for questions (or 'back' to return):")
    context = input().strip()
    
    if context.lower() == 'back':
        return
    
    if not context:
        print("❌ No context provided.")
        return
    
    print(f"\n📄 Context loaded ({len(context)} characters)")
    print("Ask questions about this text (type 'done' to finish):")
    
    qa_history = []
    
    while True:
        question = input("\n🤔 Your question: ").strip()
        
        if question.lower() == 'done':
            break
        
        if not question:
            continue
        
        # Get answer
        result = processor.answer_question(question, context)
        
        print(f"\n💡 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']}")
        
        qa_history.append({
            "question": question,
            "answer": result['answer'],
            "confidence": result['confidence']
        })
    
    # Option to save Q&A session
    if qa_history:
        save_choice = input("\n💾 Save Q&A session to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            content = f"Context:\n{context}\n\nQ&A Session:\n"
            for i, qa in enumerate(qa_history, 1):
                content += f"\nQ{i}: {qa['question']}\n"
                content += f"A{i}: {qa['answer']} (Confidence: {qa['confidence']})\n"
            
            filename = processor.save_session(content, "qa_session")
            print(f"✅ Saved as: {filename}")
def handle_explanation(processor):
    """Handle explanation generation workflow"""
    print("\n🎓 GENERATE EXPLANATION")
    print("-" * 30)
    
    # Get topic
    topic = input("Enter topic to explain (or 'back' to return): ").strip()
    if topic.lower() == 'back':
        return
    
    if not topic:
        print("❌ No topic provided.")
        return
    
    # Get age
    age = input("Target age for explanation [default: 12]: ").strip()
    if not age.isdigit():
        age = "12"
    
    # Generate explanation
    print(f"\n🔄 Generating explanation of '{topic}' for age {age}...")
    explanation = processor.generate_explanation(topic, age)
    
    print(f"\n🎓 EXPLANATION:")
    print("=" * 40)
    print(explanation)
    print("=" * 40)
    
    # Option to save
    save_choice = input("\n💾 Save explanation to file? (y/n): ").strip().lower()
    if save_choice == 'y':
        content = f"Topic: {topic}\nAge: {age}\n\nExplanation:\n{explanation}"
        filename = processor.save_session(content, "explanation")
        print(f"✅ Saved as: {filename}")
if __name__ == "__main__":
    main()