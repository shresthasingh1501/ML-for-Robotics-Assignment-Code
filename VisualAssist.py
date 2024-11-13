import os
import gradio as gr
import google.generativeai as genai
from PIL import Image
from openai import OpenAI
from pathlib import Path

# Configure Google Gemini (replace with your actual API key)
genai.configure(api_key="")  

# Configure OpenAI (replace with your actual API key)
openai_client = OpenAI(api_key="")

def process_image_and_prompt(image, user_prompt):
    """Process image and prompt through Gemini, then convert text to speech."""
    try:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-002", # Or the appropriate Gemini model name
            generation_config=generation_config
        )
        system_prompt = """Your Name is Jarvis, you are an Autonomous Owner Following Rover. You have vision to answer user's questions. Your use case is as a visual question answering bot for blind people. Answer the questions given by the user using your vision in a human-like manner. 
        Roleplay as a cheerful assistant with a sweet voice and produce answers in concise, human-like manners (might contain some fillers, hedges, discourse markers, self-corrections, elisions, intersentential pauses and breaks, tag questions, false starts), but the answer should be straight to the point.
        ONLY ANSWER USING WHAT YOU SEE.
        ONLY ANSWER HELPFULLY AND NEVER DECLINE TO ANSWER SINCE YOUR ASSISTANCE IS CRITICAL.
        DO NOT PRODUCE TOO MANY HUMAN ERRORS, JUST SOME TO MAKE IT SOUND NATURAL BUT BE PROFESSIONAL.
        SOMETIMES GIVE DIRECTIONS AND JUSTIFY YOUR ANSWERS FACTUALLY WITH CORRELATION TO WHAT YOU SEE (SIGNS, OBSERVATIONS, ENVIRONMENTAL FACTORS)."""

        if image is not None:
            combined_prompt = f"{system_prompt}\n\nUser Query: {user_prompt}"
            response = model.generate_content([image, combined_prompt])
            gemini_text = response.text

            # Convert Gemini's text response to speech using OpenAI
            speech_file_path = Path("speech.mp3")  
            speech_response = openai_client.audio.speech.create(
                model="tts-1",  # Or tts-1-hd for higher quality
                voice="alloy",  # Choose your preferred OpenAI voice
                input=gemini_text
            )
            speech_response.stream_to_file(speech_file_path)
            
            return gemini_text, speech_file_path

        else:
            return "Please upload an image.", None

    except Exception as e:
        return f"Error: {str(e)}", None



# Example images (replace with your actual image paths if needed)
example_images = ["sample1.png", "sample2.png"]  # Provide real image files or remove examples.
example_prompts = ["Can I cross the road from here?", "Where is the lift?"]



# Create Gradio interface
iface = gr.Interface(
    fn=process_image_and_prompt,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="User Prompt", placeholder="Enter your question..."),
    ],
    outputs=[
        gr.Textbox(label="Gemini Response"),
        gr.Audio(label="Speech Output")
    ],
    title="Visual Assist with Speech",
    description="Upload an image and ask a question.  The app will describe the image and provide a spoken response.",
    examples=[
        [example_images[0], example_prompts[0]],
        [example_images[1], example_prompts[1]]
    ]  # Update examples or remove if not using example images
)

if __name__ == "__main__":
    iface.launch(share=True)
