# Summarization-QA
All the files in the drive: https://drive.google.com/drive/u/2/folders/1yfD42oRpVEcuerTF1HpZHJ8pZWmpuQxJ  

This project implements an end-to-end NLP solution combining automatic text summarization and question answering capabilities using state-of-the-art transformer models. Built with Hugging Face Transformers and LangChain, it processes large text documents, generates concise summaries, and answers user queries with high accuracy.  

🚀 Features  
📑 Document Loading – Read .txt files into structured Document objects.  
  
✂️ Smart Chunking – Split long text into token-based segments for better model performance.    
  
📝 Text Summarization – Generate human-readable summaries using the facebook/bart-large-cnn model.  
  
💬 Question Answering – Ask natural language questions and retrieve context-aware answers with deepset/roberta-base-squad2.  
  
⚡ Real-Time Interaction – Summarization and QA both run instantly after processing.  
  
🛠 Tech Stack  
Transformers (facebook/bart-large-cnn, deepset/roberta-base-squad2)  
  
LangChain for document handling  
  
PyTorch backend for model execution  
  
Token-based chunking for handling long documents  
  
📂 Project Workflow  
Load Text File  
  
The script reads .txt files and stores them as Document objects.  
  
Chunk by Tokens  
  
Splits large text into overlapping chunks to fit transformer model input limits.  
  
Summarization  
  
Each chunk is summarized individually, then combined into a final comprehensive summary.  
  
Question Answering  
   
Users input a question, and the model searches through all chunks to return the most accurate answer.  
  
