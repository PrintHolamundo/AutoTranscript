# 🎧 Whisper-File-Transcriber 🎬

A Python script (`main.py`) that uses OpenAI's **Whisper** model to automatically transcribe the most recent audio or video file added to a watch directory. It prioritizes GPU usage for speed and automatically detects the language.

---

## ✨ Features

* **Single File Processing:** Only processes the **most recently modified** file in the source directory (`AUDIOS`).
* **High Accuracy:** Uses the **`large`** Whisper model for superior language detection and transcription quality.
* **GPU Priority:** Automatically attempts to use **NVIDIA CUDA (GPU)** for fast processing, falling back to **CPU** if the GPU is unavailable.
* **Auto Language Detection:** Transcribes the audio in the language it detects (e.g., Spanish, English, Russian).
* **Organized Output:** Saves the resulting `.txt` transcription to a separate **`TRANSCRIPTIONS`** folder.

---

## 🛠️ Requirements

1.  **Python 3.10**
2.  **FFmpeg** (Must be installed and accessible via your system's PATH. Whisper requires this for handling media files like `.mp4`, `.mp3`, etc.)

### 📥 Installation

You must install the required Python libraries using `pip`:

```bash
pip install -U openai-whisper tqdm