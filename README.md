# autolyrics

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

An end-to-end **lyrics transcription** application that converts singing audio into text using **OpenAI Whisper Small**. The project includes audio preprocessing, inference with Hugging Face Transformers, and an interactive web interface built with Gradio.

---

## 🚀 Live Demo

🔗 **Hugging Face Spaces:**
**https://huggingface.co/spaces/nwesha/autolyrics**

---

## ✨ Features

* 🎤 Record audio directly from your microphone
* 📁 Upload audio files (`.wav`, `.mp3`, `.flac`)
* 🎵 Automatic lyrics transcription
* ⚡ Interactive Gradio interface
* ☁️ Deployed on Hugging Face Spaces

---

## 🛠 Tech Stack

| Category           | Technologies                      |
| ------------------ | --------------------------------- |
| Language           | Python                            |
| Deep Learning      | PyTorch                           |
| Speech Recognition | OpenAI Whisper Small              |
| Frameworks         | Hugging Face Transformers, Gradio |
| Audio Processing   | Librosa                           |
| Deployment         | Hugging Face Spaces               |

---
```
## 📸 Demo

> Replace these placeholders with screenshots of your application.


### Home Page

assets/home.png


### Transcription Output


assets/output.png

```
---

## 📂 Project Structure

AutoLyrics/
│── app.py
│── requirements.txt
│── autolyrics.ipynb
│── README.md

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/nwesha/autolyrics.git
cd autolyrics
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## 📖 Usage

1. Upload an audio file or record audio.
2. Click **Generate Lyrics**.
3. View the generated transcription.

---

## 🧠 Model

* **Base Model:** OpenAI Whisper Small
* **Framework:** Hugging Face Transformers
* **Inference:** Beam Search Decoding
* **Audio Sampling Rate:** 16 kHz

The repository also contains experiments with **Parameter-Efficient Fine-Tuning (LoRA)** for adapting Whisper to singing audio transcription.

---

## 🔮 Future Work

* Improve transcription accuracy for singing audio
* Support multilingual lyrics transcription
* Add timestamped lyrics
* Optimize inference latency

---

## 👩‍💻 Author

**Anwesha**

GitHub: https://github.com/nwesha
