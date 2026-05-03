#!/usr/bin/env python3
"""
Fast Audio Transcriber

A high-performance audio transcription tool using Vosk for speech recognition.
Optimized for speed with parallel processing and efficient audio handling.
"""

import os
import json
import time
import argparse
import numpy as np
import multiprocessing as mp
from vosk import Model, KaldiRecognizer, SetLogLevel
from tqdm import tqdm
from pydub import AudioSegment
import io
import soundfile as sf
import subprocess
import sys
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and available in PATH."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def convert_audio_ffmpeg(input_path, output_path, sample_rate=16000, channels=1):
    """Convert audio to WAV format using FFmpeg with optimized settings."""
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', input_path,  # Input file
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),  # Mono audio
            '-loglevel', 'error',  # Only show errors
            output_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def transcribe_chunk(args):
    """Transcribe a single chunk of audio."""
    chunk_path, model_path = args
    
    # Load model for this process
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    
    # Read audio data
    with sf.SoundFile(chunk_path) as f:
        audio_data = f.read(dtype='int16').tobytes()
    
    # Process audio in smaller chunks
    chunk_size = 4000
    results = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if rec.AcceptWaveform(chunk):
            result = json.loads(rec.Result())
            if 'text' in result and result['text'].strip():
                results.append(result['text'])
    
    # Get final result
    final_result = json.loads(rec.FinalResult())
    if 'text' in final_result and final_result['text'].strip():
        results.append(final_result['text'])
    
    # Clean up
    os.remove(chunk_path)
    
    return ' '.join(results)

def split_audio(input_path, chunk_duration=60):
    """Split audio into smaller chunks for parallel processing."""
    try:
        # Create temp directory
        temp_dir = os.path.join(os.path.dirname(input_path), 'temp_chunks')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert to WAV if needed
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        wav_path = os.path.join(temp_dir, f"{base_name}.wav")
        
        if not convert_audio_ffmpeg(input_path, wav_path):
            return None
        
        # Get audio duration
        with sf.SoundFile(wav_path) as f:
            duration = len(f) / f.samplerate
        
        # Split into chunks
        chunks = []
        num_chunks = max(1, int(duration / chunk_duration) + 1)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
            
            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start_time),
                '-t', str(chunk_duration),
                '-i', wav_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'error',
                chunk_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
                chunks.append(chunk_path)
            except subprocess.CalledProcessError as e:
                print(f"Error creating chunk {i}: {e}")
                continue
        
        # Clean up
        os.remove(wav_path)
        return chunks
        
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return None

def download_model(model_name):
    """Download Vosk model if not already present."""
    import urllib.request
    import zipfile
    import shutil
    
    model_urls = {
        'vosk-model-en-us-0.42-gigaspeech': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip',
        'vosk-model-small-en-us-0.15': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
    }
    
    if model_name not in model_urls:
        print(f"Error: Unknown model {model_name}")
        return None
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        zip_path = f"{model_name}.zip"
        
        try:
            # Download the model
            print(f"Downloading {model_name} (this may take a while)...")
            urllib.request.urlretrieve(model_urls[model_name], zip_path)
            
            # Extract the model
            print("Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove(zip_path)
            print(f"Model {model_name} downloaded successfully")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return None
    
    return model_path

def ensure_model(model_name):
    """Ensure the Vosk model is available."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_name}' not found at {model_path}")
        print("Please make sure the model is downloaded and in the correct location.")
        return None
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Fast audio transcription using Vosk with summarization')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output file path (default: <audio_name>_transcript.txt)')
    parser.add_argument('--model', type=str, default='vosk-model-small-en-us-0.15', 
                       help='Vosk model to use (default: vosk-model-small-en-us-0.15)')
    parser.add_argument('--processes', type=int, default=0, 
                       help='Number of processes to use (0 = auto-detect, default: 0)')
    parser.add_argument('--summary', action='store_true',
                       help='Generate a summary of the transcription')
    args = parser.parse_args()
    
    # Set up output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        output_file = f"{base_name}_transcript.txt"
    else:
        output_file = args.output
    
    print(f"Starting transcription of '{os.path.basename(args.audio_path)}'...")
    
    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        print("Error: FFmpeg is required but not found. Please install FFmpeg and add it to your system PATH.")
        print("Download FFmpeg from: https://ffmpeg.org/download.html")
        return
    
    # Check if model exists
    model_path = ensure_model(args.model)
    if not model_path:
        print(f"Error: Model '{args.model}' not found.")
        print("Please download it and place it in the current directory.")
        return
    
    start_time = time.time()
    
    # Split audio into chunks for parallel processing
    print("Preparing audio for transcription...")
    chunk_paths = split_audio(args.audio_path)
    
    if not chunk_paths:
        print("Error: Could not process audio file")
        return
    
    # Determine number of processes to use
    num_processes = args.processes if args.processes > 0 else max(1, mp.cpu_count() - 1)
    num_processes = min(num_processes, len(chunk_paths))  # Don't use more processes than chunks
    
    print(f"Transcribing {len(chunk_paths)} chunks using {num_processes} processes...")
    
    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Prepare arguments for each chunk
        chunk_args = [(path, model_path) for path in chunk_paths]
        
        # Submit all tasks
        futures = [executor.submit(transcribe_chunk, arg) for arg in chunk_args]
        
        # Process results as they complete
        for future in tqdm(futures, desc="Processing chunks"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Combine results
    transcription = ' '.join(results).strip()
    
    # Clean up temp directory
    temp_dir = os.path.join(os.path.dirname(args.audio_path), 'temp_chunks')
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Get audio duration for RTF calculation
    with sf.SoundFile(args.audio_path) as f:
        audio_duration = len(f) / f.samplerate
    
    # Save transcription
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    # Print transcription and metadata
    print("\n=== TRANSCRIPTION ===")
    print(transcription)
    print("\n=== METADATA ===")
    print(f"Processing time: {process_time:.1f} seconds")
    print(f"Real-time factor: {process_time/max(audio_duration, 0.1):.2f}x")
    print(f"Transcription saved to: {output_file}")
    
    # Generate summary if requested and transcription is not empty
    if args.summary and transcription.strip():
        try:
            # Simple extractive summarization using word frequency
            sentences = re.split(r'(?<=\w[.!?])\s+', transcription)
            
            # Basic word frequency analysis
            word_freq = defaultdict(int)
            for sentence in sentences:
                for word in re.findall(r'\w+', sentence.lower()):
                    if len(word) > 3:  # Only consider words longer than 3 characters
                        word_freq[word] += 1
            
            # Score sentences based on word frequency
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = sum(word_freq[word] for word in re.findall(r'\w+', sentence.lower()) 
                          if len(word) > 3 and word in word_freq)
                sentence_scores.append((score, i))
            
            # Get top 3 sentences
            sentence_scores.sort(reverse=True)
            top_indices = sorted([i for score, i in sentence_scores[:3]])
            final_summary = ' '.join(sentences[i] for i in top_indices if i < len(sentences))
            
            # Save summary to file
            summary_file = os.path.splitext(output_file)[0] + '_summary.txt'
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(final_summary)
            
            # Print summary
            print("\n=== SUMMARY ===")
            print(final_summary)
            print(f"\nSummary saved to: {summary_file}")
            
        except Exception as e:
            print(f"\n=== SUMMARY ERROR ===")
            print(f"Error generating summary: {e}")
    elif args.summary and not transcription.strip():
        print("\n=== SUMMARY ERROR ===")
        print("No transcription available for summarization")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
