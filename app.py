from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import random
from collections import Counter
import fitz 
import pdfplumber

try:
    print("Loading spaCy model 'en_core_web_sm'...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except Exception as e:
    print(f"Failed to load spaCy model, error: {e}")
    
app = Flask(__name__)
CORS(app)

print("Flask App is starting... Server is ready.")

def generate_mcqs(text, num_questions=10):
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 4] 

    num_questions = min(num_questions, len(sentences))
    if num_questions == 0:
        return []

    selected_sentences = random.sample(sentences, num_questions)
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN" and len(token.text) > 3]

        if not nouns:
            continue

        noun_counts = Counter(nouns)
        subject = noun_counts.most_common(1)[0][0]
        
        question_stem = sentence.replace(subject, "______", 1)
        
        question_stem = question_stem.replace("\n", " ").replace("\t", " ").strip()
        
        correct_answer_text = subject
        distractors = list(set(nouns) - {correct_answer_text})
        random.shuffle(distractors)
        
        answer_choices = [correct_answer_text]
        answer_choices.extend(distractors[:3])
        
        while len(answer_choices) < 4:
            other_words = [token.text for token in doc if token.is_alpha and token.text not in answer_choices]
            if other_words:
                answer_choices.append(random.choice(other_words))
            else:
                answer_choices.append("...")

        random.shuffle(answer_choices)

        mcqs.append({
            "question": question_stem,
            "choices": answer_choices,
            "correct_answer": correct_answer_text
        })

    return mcqs

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        pdf_bytes = file_stream.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        if text.strip(): return text
        raise Exception("PyMuPDF extracted no text")
    except Exception as e1:
        try:
            file_stream.seek(0)
            with pdfplumber.open(file_stream) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                return full_text
        except Exception as e2:
            print(f"All PDF readers failed. PyMuPDF: {e1}, pdfplumber: {e2}")
            return None

@app.route('/generate-from-file', methods=['POST'])
def generate_from_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    text_input = extract_text_from_pdf(file)
    if not text_input:
        return jsonify({"error": "Could not extract any text from the file."}), 400

    mcqs = generate_mcqs(text_input)
    if not mcqs:
        return jsonify({"error": "Could not generate questions from the text."}), 400

    return jsonify({"exam": mcqs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
