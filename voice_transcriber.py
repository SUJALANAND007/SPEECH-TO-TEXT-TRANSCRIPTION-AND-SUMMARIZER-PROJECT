#!/usr/bin/env python3
"""
Voice Transcriber

This script transcribes audio files using the Vosk speech recognition library.
It converts an MP3 file to text and saves the transcription to a text file.
"""

import os
import json
import time
import argparse
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
from tqdm import tqdm
import torch
import multiprocessing as mp
from pydub import AudioSegment
import io
import soundfile as sf
import subprocess
import tempfile
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Tuple

# Disable Vosk logging
SetLogLevel(-1)

def convert_audio_ffmpeg(input_path, output_path, sample_rate=16000, channels=1):
    """Convert audio to WAV format using FFmpeg with optimized settings."""
    try:
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',  # Only show errors
            '-i', input_path,     # Input file
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),     # Mono audio
            '-acodec', 'pcm_s16le',   # 16-bit PCM
            '-f', 'wav',              # WAV format
            '-y',                     # Overwrite output file if exists
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"Error in audio conversion: {str(e)}")
        return False

def transcribe_audio(audio_path: str, model_name: str = "vosk-model-small-en-us-0.15", 
                    summarize: bool = True) -> Tuple[str, Optional[str]]:
    """
    Transcribe an audio file using Vosk with optimized sequential processing.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Vosk model to use
        summarize (bool): Whether to generate a summary of the transcription
        
    Returns:
        Tuple[str, Optional[str]]: The transcribed text and its summary (or None if summarization was not requested)
    """
    # Audio parameters
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 4000  # Process 4 seconds at a time (in samples)
    
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    try:
        # Convert audio to optimized WAV format using FFmpeg
        print("Preparing audio for transcription...")
        if not convert_audio_ffmpeg(audio_path, temp_wav_path, SAMPLE_RATE):
            raise Exception("Audio conversion failed")
            
        # Get audio duration using FFprobe
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            temp_wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip()) if result.returncode == 0 else 0
        
        print(f"Transcribing ({duration:.1f}s audio)...")
        start_time = time.time()
        
        # Initialize model and recognizer
        model = Model(model_name=model_name)
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(True)
        
        results = []
        
        # Process audio in chunks
        with open(temp_wav_path, 'rb') as wav_file:
            wav_file.seek(44)  # Skip WAV header
            
            while True:
                data = wav_file.read(CHUNK_SIZE)
                if len(data) == 0:
                    break
                    
                # Process the chunk
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if 'text' in result and result['text'].strip():
                        results.append(result['text'])
                
                # Update progress
                elapsed = time.time() - start_time
                if duration > 0:
                    position = wav_file.tell()
                    total_size = os.path.getsize(temp_wav_path) - 44  # Exclude header
                    progress = min(100, (position / total_size) * 100) if total_size > 0 else 0
                    eta = (elapsed / progress * (100 - progress)) if progress > 0 else 0
                    sys.stdout.write(f"\rProgress: {progress:.1f}% | "
                                   f"Elapsed: {elapsed:.1f}s | "
                                   f"ETA: {eta:.1f}s")
                    sys.stdout.flush()
            
            # Get the final result
            result = json.loads(rec.FinalResult())
            if 'text' in result and result['text'].strip():
                results.append(result['text'])
        
        # Clear the progress line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        
        # Combine all results
        transcription = ' '.join(results).strip()
        
        # Print summary
        elapsed = time.time() - start_time
        if duration > 0:
            print(f"Transcription complete in {elapsed:.1f} seconds (RTF: {elapsed/max(0.1, duration):.2f}")
        
        # Generate summary if requested
        summary = None
        if summarize and transcription.strip():
            print("\nGenerating summary...")
            start_summary = time.time()
            summary = summarize_text(transcription)
            print(f"Summary generated in {time.time() - start_summary:.1f} seconds")
            
        return transcription, summary
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_wav_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {str(e)}")

def summarize_text(text: str, model_name: str = "t5-small") -> str:
    """
    Summarize the given text using T5-small model.
    
    Args:
        text (str): Text to summarize
        model_name (str): Name of the T5 model to use (default: t5-small)
        
    Returns:
        str: Summarized text
    """
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # T5 requires a specific prefix for summarization
        inputs = tokenizer("summarize: " + text, 
                          return_tensors="pt", 
                          max_length=512, 
                          truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and clean up the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return ""

def save_transcription(text, output_file):
    """Save the transcribed text to a file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Transcription saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving transcription: {str(e)}")
        return False

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and available in system PATH."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def main():
    # Import time here to avoid showing import messages
    import time
    
    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        print("Error: FFmpeg is required but not found. Please install FFmpeg and add it to your system PATH.")
        print("Download FFmpeg from: https://ffmpeg.org/download.html")
        return

    # Set up argument parser
    
    # Save transcription
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"\nTranscription complete in {process_time:.1f} seconds (RTF: {process_time/max(audio_duration, 0.1):.2f}x)")
    print(f"Transcription saved to: {output_file}")
    
    # Show first 500 characters of transcription
    preview = (transcription[:500] + '...') if len(transcription) > 500 else transcription
    print("\nPreview:")
    print(preview)

if __name__ == "__main__":
    # Required for Windows multiprocessing
    if __name__ == "__main__":
        mp.freeze_support()
        main()
