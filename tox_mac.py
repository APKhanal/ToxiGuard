import sys
print("Python path (sys.path):")
for p in sys.path:
    print(p)
    
import wave
import os
import threading
import subprocess
import time
import json

# Handle non-standard imports with error checkingy
try:
    import numpy as np
except ImportError:
    print("Error: numpy module not found. Please install with: pip install numpy")
    sys.exit(1)

try:
    import whisper
except ImportError:
    print("Error: whisper module not found. Please install with: pip install openai-whisper")
    sys.exit(1)

try:
    import soundcard as sc
except ImportError:
    print("Error: soundcard module not found. Please install with: pip install soundcard")
    sys.exit(1)

TOXIC_KEYWORDS = [
    "kill yourself", "retard", "trash", "noob", "stupid", "idiot",
    "dumb", "f***", "b****", "n****", "c****", "kys", "die"
]

class AudioRecorder:
    def __init__(self, sample_rate=44100, buffer_seconds=15):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.speaker = sc.get_microphone(sc.default_microphone().name)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ToxiGuard_Output')
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self):
        print(f"[ToxiGuard] Monitoring system audio via: {self.speaker.name}")
        print(f"[ToxiGuard] Output files will be saved to: {self.output_dir}")

    def get_last_seconds(self, seconds):
        numframes = int(self.sample_rate * seconds)
        with self.speaker.recorder(samplerate=self.sample_rate) as rec:
            data = rec.record(numframes=numframes)
        return data

    def save_to_wav(self, data, filename):
        if np.any(data):  # Check if audio isn't completely silent
            max_val = np.max(np.abs(data))
            if max_val > 0:  # Prevent division by zero
                scaled = np.int16(data / max_val * 32767)
            else:
                scaled = np.zeros_like(data, dtype=np.int16)
        else:
            scaled = np.zeros_like(data, dtype=np.int16)
        filepath = os.path.join(self.output_dir,filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(data.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(scaled.tobytes())
        print(f"[ToxiGuard] Saved audio to: {filepath}")
        return filepath

class ToxiGuardBackend:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.model = whisper.load_model("base")

    def capture_after_audio(self, duration=15, filename='after.wav'):
        print("[ToxiGuard] Capturing next 15 seconds of audio...")
        with self.recorder.speaker.recorder(samplerate=self.recorder.sample_rate) as rec:
            data = rec.record(numframes=int(self.recorder.sample_rate * duration))
        return self.recorder.save_to_wav(data, filename)

    def combine_audio(self):
        output_dir = self.recorder.output_dir
        filelist_path = os.path.join(output_dir, 'filelist.txt')
        with open(filelist_path, 'w') as f:
            f.write("file 'before.wav'\n")
            f.write("file 'after.wav'\n")

        ffmpeg_path = 'ffmpeg'
        if getattr(sys, 'frozen', False):
            ffmpeg_path = os.path.join(sys._MEIPASS, 'ffmpeg.exe')

        compound_path = os.path.join(output_dir, 'compound.wav')
        cmd = [ffmpeg_path, '-f', 'concat', '-safe', '0', '-i', filelist_path, '-c', 'copy', compound_path]
        try:
            subprocess.run(cmd, check=True)
            print(f"[ToxiGuard] Combined audio saved to: {compound_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to combine audio: {e}")
        return compound_path if os.path.exists(compound_path) else None

    def transcribe_audio(self, data):
        tmp_path = os.path.join(self.recorder.output_dir, "_temp.wav")
        self.recorder.save_to_wav(data, "_temp.wav")
        print("[ToxiGuard] Transcribing audio...")
        result = self.model.transcribe(tmp_path)
        print(f"[ToxiGuard] Transcription result: {result['text']}")
        return result['text']

    def check_toxicity(self, text):
        return [word for word in TOXIC_KEYWORDS if word.lower() in text.lower()]

    def transcribe_and_score(self, compound_path, text):
        transcription_path = os.path.join(self.recorder.output_dir, "transcription.txt")
        report_path = os.path.join(self.recorder.output_dir, "toxicity_report.json")

        with open(transcription_path, "w") as f:
            f.write(text)

        found_keywords = self.check_toxicity(text)
        report = {
            "transcription": text,
            "toxicity_score": len(found_keywords) / len(TOXIC_KEYWORDS),
            "flagged_words": found_keywords
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[ToxiGuard] Transcription saved to: {transcription_path}")
        print(f"[ToxiGuard] Toxicity report saved to: {report_path}")

    def run_monitor_loop(self, interval=5):
        self.recorder.start_recording()
        print("[ToxiGuard] Starting background monitoring...")
        print(f"[ToxiGuard] Current Working Directory: {os.getcwd()}")
        while True:
            print("[ToxiGuard] Listening...")
            buffer_data = self.recorder.get_last_seconds(15)
            transcribed_text = self.transcribe_audio(buffer_data)
            print(f"[ToxiGuard] Transcribed Text: {transcribed_text}")
            found = self.check_toxicity(transcribed_text)
            print(f"[ToxiGuard] Found toxic words: {found}")
            if found:
                print("[ToxiGuard] Toxic behavior detected! Capturing full clip...")
                self.recorder.save_to_wav(buffer_data, 'before.wav')
                self.capture_after_audio()
                compound_path = self.combine_audio()
                self.transcribe_and_score(compound_path, transcribed_text)
            print("[ToxiGuard] Loop complete. Waiting for next check...")
            time.sleep(interval)

if __name__ == '__main__':
    app = ToxiGuardBackend()
    app.run_monitor_loop()
