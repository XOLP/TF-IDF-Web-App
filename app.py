import os
import re
import math
import tempfile
import json
from collections import Counter
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session
import nltk
from nltk.corpus import stopwords
import textract
import chardet

SUPPORTED_EXTENSIONS = ('.txt', '.doc', '.docx')
PER_PAGE = 10
MAX_WORDS = 50

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('russian')) | set(stopwords.words('english'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'secret')

TEMP_DIR = Path(tempfile.gettempdir()) / "tfidf_results"
TEMP_DIR.mkdir(exist_ok=True)

def process_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = re.findall(r'\b\w+\b', text)
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]

def save_results_to_file(data):
    fd, path = tempfile.mkstemp(suffix=".json", dir=TEMP_DIR)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return os.path.basename(path)

def load_results_from_file(filename):
    path = TEMP_DIR / filename
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_text_file(file_storage):
    filename = file_storage.filename.lower()
    file_bytes = file_storage.read()
    ext = Path(filename).suffix

    if ext == '.txt':
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            encoding = chardet.detect(file_bytes)['encoding'] or 'utf-8'
            return file_bytes.decode(encoding, errors='replace')
    elif ext in ('.doc', '.docx'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            try:
                text = textract.process(tmp_path).decode('utf-8')
            except Exception as e:
                raise RuntimeError(f"Ошибка извлечения текста: {e}")
        finally:
            os.remove(tmp_path)
        return text
    else:
        raise ValueError("Поддерживаются только файлы .txt, .doc, .docx")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
            try:
                text = read_text_file(file)
            except Exception as e:
                return render_template('index.html', error=f"Ошибка чтения файла: {e}")

            words = process_text(text)
            if not words:
                return render_template('index.html', error="Файл не содержит слов для анализа.")

            total_words = len(words)
            word_counts = Counter(words)

            data = []
            for word, count in word_counts.items():
                tf = count
                idf = math.log(total_words / tf) if tf else 0
                data.append({
                    'word': word,
                    'tf': tf,
                    'idf': idf  # округлять лучше при отображении
                })

            sorted_data = sorted(data, key=lambda x: (x['idf'], x['word']))

            filename = save_results_to_file(sorted_data)
            session['results_file'] = filename
            return redirect(url_for('results', page=1))

        return render_template('index.html', error="Пожалуйста, загрузите файл в формате .txt, .doc или .docx")

    return render_template('index.html')

@app.route('/results')
def results():
    filename = session.get('results_file')
    if not filename:
        return redirect(url_for('index'))

    words = load_results_from_file(filename)
    page = request.args.get('page', 1, type=int)
    per_page = PER_PAGE
    total = min(MAX_WORDS, len(words))
    total_pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    page_words = words[start:end]
    for w in page_words:
        w['idf'] = round(w['idf'], 4)

    return render_template(
        'results.html',
        words=page_words,
        page=page,
        total_pages=total_pages
    )

if __name__ == '__main__':
    app.run(debug=True)
