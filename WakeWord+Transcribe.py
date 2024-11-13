import gradio as gr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import openwakeword
from openwakeword.model import Model
from openai import OpenAI

# Download wake word models if necessary
try:
    Model()
except RuntimeError:
    openwakeword.utils.download_models()

# Initialize wake word model
model = Model(wakeword_models=["hey jarvis"])

# Initialize OpenAI client (Make sure your API key is set!)
client = OpenAI(api_key="")

def record_and_transcribe(duration, fs=16000):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()

    wav.write("recording.wav", fs, recording)

    predictions = model.predict_clip("recording.wav")

    detected = False
    for prediction in predictions:
        if prediction["hey jarvis"] > 0.6:
            detected = True
            break

    if detected:
        print("Hey Jarvis detected! Transcribing...")
        try:
            with open("recording.wav", "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                print("Transcription:", transcription)
                return "recording.wav", f"Hey Jarvis detected!\nTranscription: {transcription}"

        except Exception as e:
            print(f"Transcription error: {e}")
            return "recording.wav", f"Hey Jarvis detected!\nTranscription error: {e}"

    else:
        print("Hey Jarvis not detected.")
        return "recording.wav", "Hey Jarvis not detected."



iface = gr.Interface(
    fn=record_and_transcribe,
    inputs=[
        gr.Number(value=5, label="Duration (seconds)"),
    ],
    outputs=[
        gr.Audio("recording.wav", type="filepath"),
        gr.Textbox(label="Wake Word Detection and Transcription Result"),
    ],
    title="Audio Recorder with Wake Word Detection and Transcription",
    description="Record audio, detect 'Hey Jarvis', and transcribe if detected.",
)

iface.launch(share=True)
