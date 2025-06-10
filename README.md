
# ToxiGuard - Toxic Voice Communication Monitor

![ToxiGuard Banner](https://via.placeholder.com/800x200?text=ToxiGuard+Toxic+Voice+Monitoring) *(Consider adding an actual banner image here)*

ToxiGuard is an experimental background monitoring tool designed to detect and log toxic voice communication during online gaming sessions. It captures system audio (not microphone input), transcribes recent audio, and checks for abusive or toxic phrases.

## Features

✨ **Passive Background Monitoring**  
⌚ Records the last 15 seconds of system audio on loop  
🎧 Uses OpenAI Whisper to transcribe audio to text  
🚫 Ignores microphone input (only listens to speaker output)  
🚨 Detects and flags toxic/abusive phrases using trigger words  
🔍 Automatically saves flagged clips with full transcriptions and toxicity reports  

## Use Cases

- Parents monitoring online game sessions
- Schools or clubs supervising student behavior
- Solo players keeping logs of online interactions

## Installation

### Prerequisites
- Python 3.11+
- FFmpeg (must be in your PATH)

### Setup
```bash
pip install soundcard numpy openai-whisper
