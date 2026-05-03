#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download Vosk model if not present
VOSK_MODEL="vosk-model-small-en-us-0.15"
if [ ! -d "$VOSK_MODEL" ]; then
    echo "Downloading Vosk model..."
    wget https://alphacep.s3-us-west-2.amazonaws.com/models/$VOSK_MODEL.zip
    unzip $VOSK_MODEL.zip
    rm $VOSK_MODEL.zip
fi
