from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import nltk
from nltk.tokenize import sent_tokenize
import random
import torch
import fitz 
import pdfplumber

app = Flask(__name__)
CORS(app)

print("Flask App is starting...")

try:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')
    print("NLTK punkt downloaded successfully.")

    model_name = "google/flan-t5-base"

    print(f"Loading model and tokenizer for '{model_name}'...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
    
    print("Creating pipelines...")
    qg_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    distractor_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    print("Pipelines created successfully. Server is ready.")

except Exception as e:
    print(f"!!!!!!!! An error occurred during model loading: {e} !!!!!!!!")

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        print("Attempting to extract text with PyMuPDF...")
        pdf_bytes = file_stream.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        if text.strip():
            print("PyMuPDF extracted text successfully.")
            return text
        else:
            print("PyMuPDF ran but extracted no text. Trying pdfplumber.")
            raise Exception("PyMuPDF extracted no text")

    except Exception as e1:
        print(f"PyMuPDF failed: {e1}. Trying pdfplumber...")
        try:
            file_stream.seek(0)
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            print("pdfplumber extracted text successfully.")
            return text
        except Exception as e2:
            print(f"pdfplumber also failed: {e2}")
            return None

@app.route('/generate-from-file', methods=['POST'])
def generate_from_file():
    print("\n--- Received a new request for /generate-from-file ---")
    try:
        if 'file' not in request.files:
            print("Request error: No 'file' part in the request.")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            print("Request error: No file selected.")
            return jsonify({"error": "No selected file"}), 400

        print(f"Received file: {file.filename}")
        text_input = ""
        if file.filename.lower().endswith('.pdf'):
            text_input = extract_text_from_pdf(file)
        elif file.filename.lower().endswith('.txt'):
            print("Reading .txt file...")
            text_input = file.read().decode('utf-8')
            print("Read .txt file successfully.")
        else:
            print(f"Unsupported file type: {file.filename}")
            return jsonify({"error": "Unsupported file type. Please upload a .txt or .pdf file."}), 400

        if not text_input or not text_input.strip():
            print("Error: Could not extract any text from the file.")
            return jsonify({"error": "Could not extract any text from the file."}), 400

        print(f"Extracted {len(text_input)} characters. Starting sentence tokenization...")
        sentences = sent_tokenize(text_input)
        print(f"Found {len(sentences)} sentences. Processing the first 5...")

        if not sentences:
            return jsonify({"error": "Could not extract any text from the file."}), 400

        generated_qna = []
        for sentence in sentences[:10]: 
            if not sentence.strip():
                continue

            prompt_qg = f"Generate the question from the answer: {sentence}"
            question_output = qg_pipeline(prompt_qg, max_length=64, do_sample=False)
            question = question_output[0]['generated_text'].strip()

            correct_answer = sentence.strip()

            distractors = set()
            while len(distractors) < 3:
                distractor_prompt = f"Generate a false but related statement to: {correct_answer}"
                distractor_output = distractor_pipeline(distractor_prompt, max_length=64, do_sample=True, top_k=50, temperature=0.9)
                generated = distractor_output[0]['generated_text'].strip()
                if generated.lower() != correct_answer.lower() and len(generated.split()) < 30:
                    distractors.add(generated)

            all_choices = list(distractors) + [correct_answer]
            random.shuffle(all_choices)

            generated_qna.append({
                "question": question,
                "choices": all_choices,
                "correct_answer": correct_answer
            })

        return jsonify({"exam": generated_qna})

    except Exception as e:
        print(f"!!!!!!!! An unexpected error occurred in the main endpoint: {e} !!!!!!!!")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)