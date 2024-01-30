import os

import openai
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import assemblyai as aai
from textSummarization import summarize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi as ytt

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_summarization')
def text_summarization():
    return render_template('text_summarization.html')

@app.route('/video_summarization')
def video_summarization():
    return render_template('video_summarization.html')


@app.route('/text_summary', methods=['POST'])
def text_summary():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarize(text)
        return render_template('summary.html', summary=summary)
    return render_template('text_summarization.html', summary="no summary")


model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def extract_video_id(url:str):
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
    # fail?
    return None

def summarizer(script):
    # encode the text into tensor of integers using the appropriate tokenizer
    input_ids = tokenizer("summarize: " + script, return_tensors="pt", max_length=512, truncation=True).input_ids
    # generate the summarization output
    outputs = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)

    summary_text = tokenizer.decode(outputs[0])
    return(summary_text)



@app.route('/video_summary', methods=['GET','POST'])
def get_summary():
    if request.method == 'POST':
        url = request.form['video_link']
        video_id = extract_video_id(url)
        data = ytt.get_transcript(video_id, languages=['de', 'en'])

        scripts = []
        for text in data:
            for key, value in text.items():
                if (key == 'text'):
                    scripts.append(value)
        transcript = " ".join(scripts)
        summary = summarizer(transcript)
        return (summary)
    else:
        return "ERROR"


def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:

        text = recognizer.recognize_bing(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return None

UPLOADS_DIR = 'uploads'
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)
@app.route('/audio_summarization')
def audio_summary():
    return render_template('audio_summarization.html')
openai.api_key="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"
aai.settings.api_key = "308887e7d5a14d229350dbd739c00aa9"
@app.route('/audio_summary', methods=['POST'])
def summarize_audio():
    if 'audioFile' not in request.files:
        print("No file uploaded - request.files:", request.files)
        return "No file uploaded"

    audio_file = request.files['audioFile']

    # Check if file has a name
    if audio_file.filename == '':
        print("No selected file - audio_file.filename:", audio_file.filename)
        return "No selected file"

    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)
    print("File Path: ", audio_path)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)
    text = transcript.text
    print(text)
    summary = summarize(text)
    return summary

r = sr.Recognizer()
def transcribe_audio(path):
    # use the audio file as the audio source
    print("In transcribe audio: " , path)
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        text = r.recognize_google_cloud(audio_listened)
    return text


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)