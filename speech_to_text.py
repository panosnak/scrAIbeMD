import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import os
from pathlib import Path


# Load the m4a file
def convert_audio(input_file, output_file, input_format="m4a", output_format="wav"):
    filepath = output_file
    if os.path.exists(filepath): 
        print("File exists")
    else:  
        audio = AudioSegment.from_file(input_file, format=input_format)
        # Export to wav format
        audio.export(output_file, format=output_format)
        print(f"Conversion complete: {input_file} -> {output_file}")

def get_audio_type(file_name):
    # Extract the file extension and remove the dot
    return Path(file_name).suffix.lstrip('.').lower()        


def split_audio(audio_path, chunk_length_ms=30000):
    """
    Splits an audio file into smaller chunks of specified length.

    Args:
        audio_path (str): Path to the input audio file.
        chunk_length_ms (int): Length of each chunk in milliseconds.

    Returns:
        list: A list of audio chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks   

def speech_to_text(input_file, model_id="openai/whisper-large-v3"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )

    audio_path = input_file

    audio_type = get_audio_type(input_file)

    print("Starting splitting of the audio...")
    chunks = split_audio(audio_path)
    print(f"Audio split into {len(chunks)} chunks.")     

    transcriptions = []
    for i, chunk in enumerate(chunks):
        # Export each chunk as a temporary wav file
        temp_filename = f"temp_chunk_{i}.{audio_type}"
        chunk.export(temp_filename, format=audio_type)
        
        # Transcribe the chunk
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        result = pipe(temp_filename)
        transcriptions.append(result["text"])
        
        # Remove the temporary file
        os.remove(temp_filename)

    # Combine all transcriptions into a single string
    full_transcription = " ".join(transcriptions)

    return full_transcription    

def save_transcriptions(transcriptions, output_file):
    """
    Saves a list of transcriptions to a text file.

    Args:
        transcriptions (list): List of transcription strings.
        output_file (str): Path to the output .txt file.
    """
    with open(output_file, 'w') as f:
        for i, text in enumerate(transcriptions):
            f.write(f"{text}\n")
    print(f"Transcriptions saved to '{output_file}'")

    save_transcriptions(transcriptions, "result.txt")    


def save_to_file(input_file, output_file):
    try:
        with open(output_file, 'w') as file:
            file.write(input_file)
        print(f"Transcription saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving transcription: {e}") 
