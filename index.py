import os
import sys
from pathlib import Path

# Add the parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your Gradio app
from app import app

# Vercel requires a WSGI callable
# For Gradio 3.x, we need to expose the underlying FastAPI/Flask app
if hasattr(app, 'server'):
    # Gradio 3.x
    server = app.server
else:
    # Try to get the FastAPI app
    from fastapi import FastAPI
    import gradio as gr
    
    # Create a FastAPI app
    fastapi_app = FastAPI()
    
    # Mount the Gradio app
    fastapi_app = gr.mount_gradio_app(fastapi_app, app, path="/")
    server = fastapi_app
