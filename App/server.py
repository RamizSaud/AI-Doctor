from flask import Flask, request, jsonify, send_from_directory
import os, json, itertools, bisect, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from unsloth import FastLanguageModel
from accelerate import Accelerator
import accelerate
import time
from pydub import AudioSegment
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key = "KEY")

# LOADING MODEL
max_seq_length = 4096
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model_2",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# PROMPT
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Patient:
{}

### Doctor:
{}"""

# CHAT FUNCTION
def Sahara(query, history):
    inputs = tokenizer(
    [
        history + prompt.format(
            "If you are a doctor, please answer the medical questions based on the patient's description.",
            query,
            "",
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 200, use_cache = True)
    text_list = tokenizer.batch_decode(outputs)
    output = text_list[0]
    history = output[17:-15]
    response = output[output.rfind("### Doctor:") + 12:-15]
    return response, history

history = ""
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global history
    try:
        data = request.get_json()
        query = data.get("message")

        if not query:
            return jsonify({"error": "No message provided"}), 400

        # SAHARA CHAT
        response = ""
        if query == "quit":
            history = ""
        else:
            response, history = Sahara(query, history)
        
        # TTS
        audio_response = None
        if query != "quit":
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=response
            )

        audio_file_path = "output.mp3"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(audio_response.content)

        return jsonify({
            "response": response,
            "audio_url": audio_file_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/save-audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file in the request'}), 400
    file = request.files['audio']
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.webm')
    file.save(temp_path)
    filename = f"recording.mp3"
    final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    AudioSegment.from_file(temp_path).export(final_path, format="mp3")
    os.remove(temp_path)

    # STT
    audio_file = open("recording.mp3", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    transcript = transcription.text

    return jsonify({'message': 'File saved successfully', 'filename': filename, 'transcript': transcript}), 200
    
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/output.mp3')
def serve_audio():
    return send_from_directory('.', 'output.mp3')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
