import gradio as gr
import os
import subprocess
import tempfile
import sys
import json
from pathlib import Path

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check if Vosk model is available
VOSK_MODEL = "vosk-model-small-en-us-0.15"

# For Hugging Face Spaces, we'll use a different approach to handle the model
def check_vosk_model():
    """Check if Vosk model is available, use a smaller model if needed."""
    try:
        # First try to use the existing model
        if os.path.exists(VOSK_MODEL):
            return True
            
        # For Hugging Face Spaces, we'll use a smaller model
        print("Using smaller Vosk model for Hugging Face Spaces...")
        import urllib.request
        import zipfile
        
        # Use a smaller model for Hugging Face Spaces
        model_url = "https://alphacep.s3-us-west-2.amazonaws.com/models/vosk-model-small-en-us-0.15.zip"
        zip_path = f"{VOSK_MODEL}.zip"
        
        print(f"Downloading {VOSK_MODEL}...")
        urllib.request.urlretrieve(model_url, zip_path)
        
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(zip_path)
        print("Vosk model is ready!")
        return True
        
    except Exception as e:
        print(f"Error with Vosk model: {e}")
        if 'zip_path' in locals() and os.path.exists(zip_path):
            os.remove(zip_path)
        return False

def generate_ai_summary(text, max_length=150, min_length=30):
    """
    Generate a summary of the given text using a pre-trained T5 model.
    
    Args:
        text (str): Input text to summarize
        max_length (int): Maximum length of the summary
        min_length (int): Minimum length of the summary
        
    Returns:
        str: Generated summary
    """
    try:
        # First try using the smaller T5 model
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        import re
        
        # Clean and preprocess the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Load the model and tokenizer
        model_name = "t5-small"  # Much smaller and faster than BART
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # Prepare the input text with summarization prompt
        inputs = ["summarize: " + text]
        
        # Tokenize the input
        inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt").to(device)
        
        # Generate the summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and clean up the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return summary
        
    except Exception as e:
        print(f"Error in T5 summarization: {str(e)}")
        
        # Fallback to extractive summarization if AI summarization fails
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            
            # Initialize the parser and tokenizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            
            # Initialize the TextRank summarizer (better than LSA for short texts)
            summarizer = TextRankSummarizer()
            
            # Generate the summary (3 sentences)
            summary_sentences = summarizer(parser.document, 3)
            
            # Join the summary sentences
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            
            # Clean up the summary
            summary = re.sub(r'\s+', ' ', summary).strip()
            if summary and summary[-1] not in '.!?':
                summary += '.'
                
            return summary
            
        except Exception as e2:
            print(f"Fallback summarization also failed: {str(e2)}")
            # Last resort: return the first few sentences
            sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            return '. '.join(sentences[:3]) + '.'

def transcribe(audio_file, generate_summary=False):
    """
    Transcribe an audio file and optionally generate a summary.
    
    Args:
        audio_file (str): Path to the audio file
        generate_summary (bool): Whether to generate a summary
        
    Returns:
        dict: Dictionary containing 'transcription' and optionally 'summary'
    """
    # Initialize result dictionary
    result_dict = {
        "transcription": "",
        "summary": "",
        "transcript_file": "",
        "summary_file": ""
    }
    
    # Create a temporary file for the transcript
    base_name = os.path.splitext(os.path.basename(audio_file))[0] if audio_file else "transcript"
    temp_path = f"{base_name}_transcript.txt"
    
    try:
        # Call the fast_transcriber.py script
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "fast_transcriber.py",
            "--model", "vosk-model-small-en-us-0.15",
            "--output", "transcription_output.txt"
        ]
        
        if audio_file:
            cmd.append(audio_file)
        
        # Run the transcription
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = f"Transcription failed with return code {result.returncode}. "
            error_msg += f"Error: {result.stderr}" if result.stderr else "No error details available."
            raise Exception(error_msg)
        
        # Read the output file
        if not os.path.exists("transcription_output.txt"):
            raise FileNotFoundError("Transcription output file was not created.")
            
        with open("transcription_output.txt", "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError("Transcription resulted in empty content.")
        
        # Save the transcription to the result
        result_dict["transcription"] = content
        
        # Save the transcription to a file
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        result_dict["transcript_file"] = os.path.abspath(temp_path)
        
        # Generate summary if requested
        if generate_summary and content:
            print("Generating summary...")
            summary = generate_ai_summary(content)
            result_dict["summary"] = summary
            
            # Save the summary to a file
            summary_path = f"{base_name}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            result_dict["summary_file"] = os.path.abspath(summary_path)
        
        return result_dict
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred during transcription: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        # Initialize the result dictionary with error message
        result_dict["transcription"] = f"Error during transcription: {str(e)}"
        result_dict["summary"] = "An error occurred during processing. Please check the console for details."
        
        # Try to read from the output file as a last resort
        if os.path.exists(temp_path):
            try:
                with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                    result_dict["transcription"] = f.read().strip()
            except Exception as read_error:
                print(f"Error reading temp file: {read_error}")
        
        return result_dict
    
    finally:
        # Clean up any temporary files if needed
        pass

def process_audio(audio_file, generate_summary):
    """
    Process an audio file to transcribe and optionally summarize it.
    
    Args:
        audio_file (str): Path to the audio file to process
        generate_summary (bool): Whether to generate a summary
        
    Returns:
        tuple: (transcription, summary, show_transcript_btn, show_summary_btn, show_summary_col)
    """
    if audio_file is None or not os.path.exists(audio_file):
        return "Please upload a valid audio file.", "", False, False, False
    
    # Show a processing message
    print(f"Starting transcription process for {audio_file}...")
    
    try:
        # Create the transcripts directory if it doesn't exist
        os.makedirs("transcripts", exist_ok=True)
        
        # Get the base name for output files
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        transcript_file = os.path.abspath(os.path.join("transcripts", f"{base_name}_transcript.txt"))
        summary_file = os.path.abspath(os.path.join("transcripts", f"{base_name}_summary.txt"))
        
        # Call the transcription function with error handling
        try:
            result = transcribe(audio_file, generate_summary)
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "Error: Could not transcribe the audio.", False, False, True
        
        # Check if we got a valid result
        if not result or not isinstance(result, dict) or "transcription" not in result:
            error_msg = "No valid transcription result returned from the transcription service."
            print(error_msg)
            if result:
                print(f"Raw result: {result}")
            return error_msg, "Error: Could not transcribe the audio.", False, False, True
        
        # Get the transcription text
        transcription = str(result.get("transcription", "")).strip()
        
        # Handle empty transcription
        if not transcription or transcription.lower() == "none":
            transcription = "No speech was detected in the audio file. It may be too short, silent, or inaudible."
        
        # Save the transcription to a file
        transcript_saved = False
        try:
            with open(transcript_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(transcription)
            print(f"Transcript saved to: {transcript_file}")
            transcript_saved = True
        except Exception as e:
            print(f"Warning: Could not save transcript to file: {e}")
        
        # Handle the summary
        has_summary = False
        summary_text = ""
        
        if generate_summary:
            summary_text = str(result.get("summary", "")).strip()
            
            # If we have a summary, save it to a file
            if summary_text and summary_text.lower() not in ["none", "summary generation was not requested or failed."]:
                has_summary = True
                try:
                    with open(summary_file, 'w', encoding='utf-8', errors='replace') as f:
                        f.write(summary_text)
                    print(f"Summary saved to: {summary_file}")
                except Exception as e:
                    print(f"Warning: Could not save summary to file: {e}")
                    has_summary = False
            else:
                # If summary generation was requested but failed, provide a helpful message
                summary_text = "Summary generation was not successful. The audio might be too short or not contain enough content."
        else:
            summary_text = ""
            
            # Save the summary to a file if we have one
            try:
                with open(summary_file, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(summary_text)
                print(f"Summary saved to: {summary_file}")
            except Exception as e:
                print(f"Warning: Could not save summary to file: {e}")
                has_summary = False
        
        # Log completion
        print("Transcription completed successfully!")
        
        # Debug output to check summary generation
        print(f"Summary requested: {generate_summary}")
        print(f"Has summary: {has_summary}")
        print(f"Summary text length: {len(summary_text) if summary_text else 0}")
        
        # Show download buttons if we have content
        show_transcript_btn = transcript_saved and os.path.exists(transcript_file) and os.path.getsize(transcript_file) > 0
        show_summary_btn = has_summary and os.path.exists(summary_file) and os.path.getsize(summary_file) > 0
        
        print(f"Process complete - Show buttons - Transcript: {show_transcript_btn}, Summary: {show_summary_btn}")
        
        # Return the results
        return (
            transcription,
            summary_text if has_summary else ("" if not generate_summary else "Summary generation was not successful."),
            show_transcript_btn,  # Show download transcript button if we have a transcript
            show_summary_btn,     # Show download summary button if we have a summary
            True                  # Always show the summary column
        )
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred during processing: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)  # Print full error for debugging
        
        # Try to provide a more user-friendly error message
        user_error = str(e)
        if "No such file or directory" in user_error:
            user_error = "Error: Could not access the audio file. Please try uploading it again."
        elif "format not supported" in user_error.lower():
            user_error = "Error: The audio format is not supported. Please try with a different file format (e.g., .wav, .mp3)."
        
        # Return error state
        return (
            f"Error: {user_error}",
            "An error occurred during processing. Please try again or check the console for details.",
            False,  # Hide download buttons
            False,
            True    # Keep summary column visible
        )

# Custom CSS for modern UI/UX
custom_css = """
:root {
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --secondary: #f3f4f6;
    --text: #1f2937;
    --text-light: #6b7280;
    --background: #ffffff;
    --card-bg: #f9fafb;
    --border: #e5e7eb;
    --success: #10b981;
    --error: #ef4444;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    color: var(--text);
    background-color: var(--background);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.title {
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, var(--primary), #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 1.125rem;
    color: var(--text-light);
    max-width: 600px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}

.card {
    background: var(--card-bg);
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    margin-bottom: 2rem;
    border: 1px solid var(--border);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.25rem;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.btn {
    background: var(--primary);
    color: white !important;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.btn:hover {
    background: var(--primary-hover);
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.btn:active {
    transform: translateY(0);
}

.btn-secondary {
    background: var(--secondary);
    color: var(--text) !important;
}

.btn-secondary:hover {
    background: #e5e7eb;
}

.btn-icon {
    width: 1.25rem;
    height: 1.25rem;
}

.input-group {
    margin-bottom: 1.5rem;
}

.label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: var(--text);
}

.checkbox-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1rem 0;
}

.checkbox {
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 0.375rem;
    border: 1px solid var(--border);
}

.textarea {
    width: 100%;
    min-height: 150px;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border);
    font-family: inherit;
    font-size: 0.9375rem;
    line-height: 1.5;
    resize: vertical;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
}

.status {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status.success {
    background-color: #ecfdf5;
    color: var(--success);
    border: 1px solid #a7f3d0;
}

.status.error {
    background-color: #fef2f2;
    color: var(--error);
    border: 1px solid #fecaca;
}

.download-buttons {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.tabs {
    margin-top: 1.5rem;
}

.tab-buttons {
    display: flex;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.tab-button {
    padding: 0.75rem 1.5rem;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 500;
    color: var(--text-light);
    cursor: pointer;
    transition: all 0.2s ease;
}

.tab-button.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
    color: var(--text-light);
    font-size: 0.875rem;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --text: #f3f4f6;
        --text-light: #9ca3af;
        --background: #111827;
        --card-bg: #1f2937;
        --border: #374151;
        --secondary: #374151;
    }
    
    .btn-secondary {
        color: var(--text) !important;
    }
    
    .textarea {
        background-color: #1f2937;
        color: var(--text);
        border-color: var(--border);
    }
}
"""

# Disable Gradio analytics
import gradio as gr

gr.close_all()  # Close any existing Gradio instances

# Create the Gradio interface with enhanced UI
with gr.Blocks(
    title="Audio Transcriber Pro",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="indigo",
        neutral_hue="slate",
        spacing_size="sm",
        radius_size="md",
        font=["Inter", "sans-serif"],
    )
) as app:
    # Header Section
    with gr.Row(elem_classes="header"):
        gr.Markdown("""
        <div class="container">
            <h1 class="title">Audio Transcriber Pro</h1>
            <p class="subtitle">
                Upload an audio file to get an accurate transcription and AI-powered summary instantly.
                Perfect for meetings, interviews, lectures, and more.
            </p>
        </div>
        """)
    
    # Main Content
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            # Upload Section
            gr.Markdown("### 🎙️ Upload Audio")
            audio_input = gr.Audio(
                label="Drag and drop your audio file here or click to browse",
                type="filepath"
            )
            
            with gr.Row():
                transcribe_btn = gr.Button(
                    "🎤 Transcribe Audio",
                    variant="primary"
                )
                clear_btn = gr.Button(
                    "🗝️ Clear",
                    variant="secondary"
                )
            
            with gr.Row():
                summarize_cb = gr.Checkbox(
                    label="✨ Generate AI Summary",
                    value=True
                )
                
                gr.Markdown("""
                <div style="text-align: right; margin-top: 8px;">
                    <small>Supports: MP3, WAV, M4A</small>
                </div>
                """)
            
            status = gr.Textbox(label="Status", interactive=False, visible=False)
        
        # Right Column - Output
        with gr.Column(scale=2):
            # Tabs for better organization
            with gr.Tabs():
                # Transcription Tab
                with gr.TabItem("📝 Transcription"):
                    transcription_output = gr.Textbox(
                        label="Transcription",
                        placeholder="Your transcription will appear here...",
                        lines=12,
                        max_lines=20,
                        interactive=False
                    )
                    # Transcript download row - initially visible
                    with gr.Row() as transcript_btn_row:
                        download_transcript_btn = gr.Button(
                            "💾 Download Transcript",
                            variant="secondary",
                            visible=False  # Will be shown when transcription is complete
                        )
            
                # Summary Tab
                with gr.TabItem("📋 Summary"):
                    summary_output = gr.Textbox(
                        label="AI-Generated Summary",
                        placeholder="Your summary will appear here...",
                        lines=8,
                        max_lines=15,
                        interactive=False
                    )
                    # Summary download row - initially visible
                    with gr.Row() as summary_btn_row:
                        download_summary_btn = gr.Button(
                            "💾 Download Summary",
                            variant="secondary",
                            visible=False  # Will be shown when summary is available
                        )
    
    # Footer
    with gr.Row():
        gr.Markdown("""
        <div class="footer">
            <p>Audio Transcriber Pro | Built with ❤️ using Gradio</p>
            <p>Supports multiple languages and audio formats</p>
        </div>
        """)
    
    # Set up event handlers
    def on_transcribe_click(audio_file, generate_summary, progress=gr.Progress()):
        """
        Handle the transcribe button click event.
        
        Args:
            audio_file (str): Path to the audio file to transcribe
            generate_summary (bool): Whether to generate a summary
            
        Returns:
            tuple: Tuple containing (transcription, summary, show_transcript_btn, show_summary_btn, show_summary_col, status_update)
        """
        if not audio_file:
            return "Please upload an audio file first.", "", False, False, True, gr.update(visible=True)
        
        try:
            # Show processing status
            progress(0.1, desc="🔍 Starting transcription...")
            
            # Process the audio file
            progress(0.3, desc="🎤 Transcribing audio...")
            transcription, summary, show_transcript, show_summary, show_summary_col = process_audio(audio_file, generate_summary)
            
            # Check if the transcription is empty or too short
            if len(transcription.strip()) < 10:  # Arbitrary threshold for very short transcriptions
                status_msg = "⚠️ Warning: The transcription is very short. The audio might be too quiet or inaudible."
                print(status_msg)
                return "", "", False, False, True, gr.update(visible=True, value=status_msg)
            
            # Generate summary if requested
            if generate_summary:
                progress(0.7, desc="📝 Generating summary...")
                # The summary is already generated in process_audio, we just need to update the status
            
            # Success case
            progress(0.9, desc="✅ Processing complete!")
            status_msg = "✅ Transcription completed successfully!"
            
            # Update status message if summary was generated
            if generate_summary and show_summary and summary and "error" not in summary.lower():
                status_msg += " Summary generated."
            
            print(status_msg)
            
            # Small delay to show completion
            import time
            time.sleep(0.5)
            
            return (
                transcription, 
                summary, 
                gr.update(visible=show_transcript), 
                gr.update(visible=show_summary),
                gr.update(visible=show_summary_col),
                gr.update(visible=True, value=status_msg)  # Show success status
            )
            
        except Exception as e:
            import traceback
            error_msg = f"An error occurred: {str(e)}"
            print(f"Error in transcription: {error_msg}")
            traceback.print_exc()
            # Provide user-friendly error messages
            if "No such file or directory" in str(e):
                error_msg = "❌ Error: Could not access the audio file. Please try uploading it again."
            elif "format not supported" in str(e).lower():
                error_msg = "❌ Error: The audio format is not supported. Please try with a different file format (e.g., .wav, .mp3)."
            else:
                error_msg = f"❌ An unexpected error occurred: {str(e)}. Please check the console for details."
            
            return (
                "An error occurred during transcription. Please check the console for details.",
                "",
                False,
                False,
                True,
                gr.update(visible=True, value=error_msg)
            )
    
    # Connect the transcribe button
    transcribe_btn.click(
        fn=on_transcribe_click,
        inputs=[audio_input, summarize_cb],
        outputs=[
            transcription_output,
            summary_output,
            download_transcript_btn,
            download_summary_btn,
            status
        ]
    )
    
    # Set up download handlers
    def get_transcript_path(audio_file):
        if not audio_file:
            return None
            
        try:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            transcript_path = os.path.abspath(os.path.join("transcripts", f"{base_name}_transcript.txt"))
            
            # Ensure the file exists
            if not os.path.exists(transcript_path):
                print(f"Transcript file not found: {transcript_path}")
                # Try to create an empty file if it doesn't exist
                try:
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write("")
                    print(f"Created empty transcript file: {transcript_path}")
                except Exception as e:
                    print(f"Failed to create transcript file: {e}")
                    return None
                
            print(f"Transcript path: {transcript_path}")
            return transcript_path
        except Exception as e:
            print(f"Error getting transcript path: {e}")
            return None
    
    def get_summary_path(audio_file):
        if not audio_file:
            return None
            
        try:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            summary_path = os.path.abspath(os.path.join("transcripts", f"{base_name}_summary.txt"))
            
            # Ensure the file exists
            if not os.path.exists(summary_path):
                print(f"Summary file not found: {summary_path}")
                # Try to create an empty file if it doesn't exist
                try:
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write("")
                    print(f"Created empty summary file: {summary_path}")
                except Exception as e:
                    print(f"Failed to create summary file: {e}")
                    return None
                
            print(f"Summary path: {summary_path}")
            return summary_path
        except Exception as e:
            print(f"Error getting summary path: {e}")
            return None
    
    # Connect download buttons with proper file handling
    def download_file(file_path):
        if not file_path or not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        print(f"Downloading file: {file_path}")
        return file_path
    
    def get_download_paths(audio_file, generate_summary):
        """Return both transcript and summary paths for debugging"""
        transcript_path = get_transcript_path(audio_file) if audio_file else None
        summary_path = get_summary_path(audio_file) if audio_file and generate_summary else None
        print(f"Download paths - Transcript: {transcript_path}, Summary: {summary_path}")
        return transcript_path, summary_path
    
    # Update download buttons when transcription completes
    def update_download_buttons(audio_file, generate_summary):
        if not audio_file:
            return False, False
            
        transcript_path = get_transcript_path(audio_file)
        summary_path = get_summary_path(audio_file) if generate_summary else None
        
        # Check if files exist and have content
        transcript_exists = transcript_path and os.path.exists(transcript_path) and os.path.getsize(transcript_path) > 0
        summary_exists = summary_path and os.path.exists(summary_path) and os.path.getsize(summary_path) > 0
        
        print(f"Updating download buttons - Transcript: {transcript_exists}, Summary: {summary_exists}")
        return transcript_exists, summary_exists
    
    # Connect the download buttons
    download_transcript_btn.click(
        fn=get_transcript_path,
        inputs=[audio_input],
        outputs=gr.File(label="Download Transcription")
    )
    
    download_summary_btn.click(
        fn=get_summary_path,
        inputs=[audio_input],
        outputs=gr.File(label="Download Summary")
    )
    
    # Update button visibility when transcription completes
    transcribe_btn.click(
        fn=update_download_buttons,
        inputs=[audio_input, summarize_cb],
        outputs=[download_transcript_btn, download_summary_btn]
    )

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("transcripts", exist_ok=True)
    
    # Configure Gradio app settings and disable warnings
    import warnings
    import gradio as gr
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
    
    # Queue configuration (for handling multiple requests)
    # Using the simplest queue configuration that's compatible with most Gradio versions
    app.queue()
    
    # Set up server configuration with minimal parameters for compatibility
    server_name = "127.0.0.1"
    server_port = 7860
    
    print("Starting Audio Transcriber application...")
    print(f"Server will be available at: http://{server_name}:{server_port}")
    
    # Get Gradio version for debugging
    import gradio as gr
    print(f"Gradio version: {gr.__version__}")
    
    # Launch the app with minimal configuration
    try:
        # First try with queue enabled (for newer Gradio versions)
        try:
            app.queue()  # Enable queue for better handling of multiple requests
            app.launch(server_name=server_name, server_port=server_port, share=False)
        except (TypeError, AttributeError):
            # Fallback if queue is not supported
            app.launch(server_name=server_name, server_port=server_port, share=False)
    except Exception as e:
        print(f"Error launching app: {str(e)}")
        print("Trying with absolute minimal configuration...")
        # Try with absolute minimal configuration
        app.launch(server_port=server_port)
