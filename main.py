import whisper
import os
import glob
import platform
import torch # Import PyTorch to better check for CUDA availability

# --- Configuration ---
AUDIO_DIR = "AUDIOS"
TRANSCRIPT_DIR = "TRANSCRIPTIONS"
VALID_EXTENSIONS = ('.mp3', '.mp4', '.wav', '.flac', '.m4a', '.ogg', '.mov', '.avi')
WHISPER_MODEL = "medium"

# --- Device Detection Logic ---
if platform.system() == 'Darwin':
    # 'Darwin' is the kernel name for macOS. 'mps' enables GPU acceleration on Apple Silicon (M1/M2/M3).
    DEVICE = "mps"
    print("💡 Detected macOS (Darwin). Attempting to use 'mps' for GPU acceleration.")
elif platform.system() == 'Linux' or platform.system() == 'Windows':
    # Check for CUDA availability more accurately using PyTorch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print("💡 Detected CUDA-enabled GPU (NVIDIA). Attempting to use 'cuda'.")
    else:
        # Fallback to CPU if PyTorch is installed but CUDA is not available/configured
        DEVICE = "cpu"
        print("💡 Detected Linux/Windows, but no CUDA device found. Using 'cpu'.")
else:
    # Default for any other unknown system
    DEVICE = "cpu"
    print("💡 Using 'cpu' by default for an unrecognized operating system.")
# --- End Device Detection ---


def find_latest_file(directory):
    """Finds the most recently modified audio/video file in the directory."""

    # 1. Get all files matching valid extensions
    all_files = []
    for ext in VALID_EXTENSIONS:
        all_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))

    if not all_files:
        return None

    # 2. Return the file with the newest modification time (getmtime)
    return max(all_files, key=os.path.getmtime)


def transcribe_latest_file():
    """Main function to load the model and process the newest file."""

    # --- 1. Setup Directories ---
    for directory in [AUDIO_DIR, TRANSCRIPT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: **{directory}**")

    # --- 2. Find Target File ---
    latest_file_path = find_latest_file(AUDIO_DIR)

    if latest_file_path is None:
        print(f"No valid files found in **{AUDIO_DIR}**.")
        return

    file_name = os.path.basename(latest_file_path)
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.txt")
    print(f"\n✅ Found latest file: **{file_name}**")

    # --- 3. Load Whisper Model ---
    try:
        # Use the automatically determined DEVICE variable
        print(f"Loading **{WHISPER_MODEL}** model on **{DEVICE}**...")
        model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    except Exception as e:
        # Fallback to CPU if the determined device (mps/cuda) fails to load
        print(f"Failed to load '{DEVICE}': {e}. Falling back to 'cpu'.")
        try:
            model = whisper.load_model(WHISPER_MODEL, device="cpu")
        except Exception as cpu_e:
            print(f"Failed to load model on CPU: {cpu_e}")
            return
    print("Model loaded successfully.")

    # --- 4. Transcribe and Save ---
    print(f"\n🎬 Starting transcription for: **{file_name}**")

    try:
        # Transcribe the audio file
        result = model.transcribe(latest_file_path, verbose=True)

        transcript_text = result["text"]
        detected_language = result["language"]

        # Save to the TRANSCRIPTIONS folder
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"--- Detected Language (Model {WHISPER_MODEL}) on {DEVICE.upper()}: {detected_language.upper()} ---\n\n")
            f.write(transcript_text)

        print(f"\n🎉 Successfully saved transcript in **{TRANSCRIPT_DIR}** as: **{os.path.basename(output_path)}**")
        print(f"   Language Detected: **{detected_language.upper()}**")

    except Exception as e:
        print(f"❌ Error during processing: {e}")


if __name__ == "__main__":
    transcribe_latest_file()