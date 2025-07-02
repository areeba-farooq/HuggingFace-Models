# LangChain + HuggingFace + Transformers Practice

A comprehensive practice project demonstrating the integration of LangChain with HuggingFace models using the Transformers library. This project showcases text summarization, question-answering, and text generation capabilities.

## 🚀 Features

- **Text Summarization**: Intelligent text summarization with customizable length
- **Question Answering**: Interactive Q&A system based on summarized content
- **Text Generation**: Educational content generation with age-appropriate explanations
- **Pipeline Integration**: Seamless integration between HuggingFace pipelines and LangChain chains

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd HuggingFace-Models
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- `transformers` - HuggingFace Transformers library for pre-trained models
- `langchain` - Framework for developing applications with language models
- `langchain-huggingface` - LangChain integration with HuggingFace models

## 🤖 Models Used

### 1. Text Summarization
- **Model**: `facebook/bart-large-cnn`
- **Purpose**: Extractive and abstractive text summarization
- **Description**: BART (Bidirectional and Auto-Regressive Transformers) fine-tuned on CNN/DailyMail dataset for news summarization

### 2. Question Answering
- **Model**: `distilbert-base-uncased-distilled-squad`
- **Purpose**: Reading comprehension and question answering
- **Description**: Distilled version of BERT, fine-tuned on SQuAD dataset, optimized for speed while maintaining accuracy

### 3. Text Generation
- **Model**: `gpt2`
- **Purpose**: Educational content generation and text completion
- **Description**: Generative Pre-trained Transformer 2, capable of generating coherent and contextually relevant text

## 📁 Project Structure

```
langchain-huggingface-practice/
├── README.md
├── requirements.txt
├── app.py              # Advanced summarization + Q&A application
└── main.py             # Basic text generation with LangChain
```

## 🎯 Usage

### Advanced Application (app.py)

Run the comprehensive summarization and question-answering application:

```bash
python app.py
```

**Features:**
- Interactive text input for summarization
- Customizable summary length (short/medium/long)
- Follow-up question answering based on the summary
- Continuous Q&A session until user exits

**Example Workflow:**
1. Enter text to summarize
2. Choose summary length
3. Review generated summary
4. Ask questions about the summary
5. Type 'exit' to end the session

### Basic Text Generation (main.py)

Run the educational text generation example:

```bash
python main.py
```

**Features:**
- Topic-based explanation generation
- Age-appropriate content adaptation
- LangChain template integration

**Example Usage:**
- Enter topic: "Machine Learning"
- Enter age: "12"
- Get age-appropriate explanation of the topic

## 🔧 Code Highlights

### LangChain Pipeline Integration
```python
from langchain_huggingface import HuggingFacePipeline

# Wrap HuggingFace pipeline as LangChain model
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)
```

### Prompt Template Usage
```python
from langchain.prompts import PromptTemplate

# Create reusable prompt templates
template = PromptTemplate.from_template(
    "Explain {topic} in detail for a {age} year old to understand."
)
```

### Chain Creation
```python
# Combine template and model into a chain
chain = template | llm
response = chain.invoke({"topic": topic, "age": age})
```

## ⚡ Performance Optimizations

- **Warning Suppression**: Transformers warnings are suppressed for cleaner output
- **GPU Support**: Automatic GPU utilization when available (`device=0`)
- **Model Efficiency**: Uses DistilBERT for faster question answering
- **Memory Management**: Proper truncation and length limits for text generation

## 🎓 Learning Objectives

This project demonstrates:

1. **Model Integration**: How to integrate HuggingFace models with LangChain
2. **Pipeline Chaining**: Creating complex workflows with multiple models
3. **Interactive Applications**: Building user-friendly CLI applications
4. **Prompt Engineering**: Using templates for consistent and effective prompts
5. **Multi-Modal AI**: Combining different AI capabilities (summarization, Q&A, generation)

## 🔮 Potential Extensions

- Add support for larger models (Mistral-7B, Llama-2)
- Implement web interface using Flask/FastAPI
- Add document upload and processing capabilities
- Integrate vector databases for semantic search
- Add model fine-tuning examples
- Implement streaming responses for better UX

## 📝 Notes

- Models are downloaded automatically on first run
- GPU acceleration is enabled if CUDA is available
- Text generation is limited to 256 tokens for demonstration purposes
- The project uses smaller models for faster execution and lower resource usage

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or additional features!

## 📄 License

This project is open source and available under the MIT License.
