from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import pipeline, AutoTokenizer
from langchain.schema import Document
import os

app = Flask(__name__)

# Initialize models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Store uploaded text and chunks globally (for simplicity; use a database for production)
uploaded_text = ""
text_chunks = []

def load_txt_as_document(text_content):
    return [Document(page_content=text_content, metadata={"source": "uploaded_text"})]

def chunk_by_tokens(text, max_tokens=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk, clean_up_tokenization_spaces=True))
        start += max_tokens - overlap
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_text, text_chunks
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and file.filename.endswith('.txt'):
        uploaded_text = file.read().decode('utf-8')
        docs = load_txt_as_document(uploaded_text)
        text_chunks = chunk_by_tokens(docs[0].page_content)
        summaries = []
        for chunk in text_chunks:
            result = summarizer(chunk, max_length=200, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
        final_summary = " ".join(summaries)
        return jsonify({"summary": final_summary})
    return jsonify({"error": "Invalid file format, only .txt allowed"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global text_chunks
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not text_chunks:
        return jsonify({"error": "No text uploaded yet"}), 400
    best_answer = None
    best_score = 0
    for chunk in text_chunks:
        result = qa_model(question=question, context=chunk)
        if result['score'] > best_score:
            best_score = result['score']
            best_answer = result['answer']
    return jsonify({"answer": best_answer, "confidence": best_score})

if __name__ == '__main__':
    app.run(debug=True)