import torch
import librosa
import gradio as gr

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from peft import PeftModel


# Configuration
MODEL_NAME = "openai/whisper-small"
ADAPTER_PATH = "./whisper-lora-autolyrics"

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load processor
processor = WhisperProcessor.from_pretrained(
    ADAPTER_PATH
)


model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("AutoLyrics model loaded successfully!")


def transcribe_audio(audio_file):

    if audio_file is None:
        return "No audio uploaded."

    speech_array, _ = librosa.load(
        audio_file,
        sr=16000,
        mono=True
    )

    input_features = processor.feature_extractor(
        speech_array,
        sampling_rate=16000
    ).input_features[0]

    input_tensor = torch.tensor(
        [input_features],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        predicted_ids = model.generate(
            input_features=input_tensor,
            max_new_tokens=225,
            num_beams=3
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    print(f"Audio length: {len(speech_array)/16000:.2f} seconds")
    print(transcription)
    return transcription


theme = gr.themes.Soft()

with gr.Blocks() as demo:

    gr.Markdown("""
    # AutoLyrics

    ### Singing Lyrics Transcription using Fine-Tuned Whisper

    Upload or record a song and generate lyrics automatically.
    
    Developed By: Apoorva, Anwesha, Sujal
    """)

    with gr.Accordion("Model Information", open=False):

        gr.Markdown("""
        **Base Model:** OpenAI Whisper Small
        **Fine-Tuning:** LoRA
        **Task:** Lyrics Transcription from Singing Audio
        **Sampling Rate:** 16 kHz
        **Decoding:** Beam Search
        """)

    with gr.Row():

        with gr.Column():

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload or Record Audio"
            )

            transcribe_btn = gr.Button(
                "Generate Lyrics",
                variant="primary"
            )

        with gr.Column():

            output_box = gr.Textbox(
                label="Generated Lyrics",
                lines=15,
            )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=output_box
    )

    gr.Markdown("""
    ---
    Built using Whisper, LoRA, PyTorch, Hugging Face Transformers, and Gradio.
    """)

demo.launch()