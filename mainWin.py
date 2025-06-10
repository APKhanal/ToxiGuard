import numpy as np
import wave
import os
import sys
import subprocess
import time
import whisper
import json

KEYWORDS_FILE = "keywords.json"

class AudioRecorder:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ToxiGuard_Output')
        os.makedirs(self.output_dir, exist_ok=True)

    def record_system_audio(self, duration_seconds=15, filename="_temp.wav"):
        filepath = os.path.join(self.output_dir, filename)
        command = [
            "ffmpeg",
            "-y",
            "-f", "dshow",
            "-i", "audio=CABLE Output (VB-Audio Virtual Cable)",
            "-t", str(duration_seconds),
            filepath
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return filepath

class ToxiGuardBackend:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.model = whisper.load_model("base")
        self.toxic_keywords = self.load_keywords()
        self.severity_threshold = 1  # default severity tolerance level

    def load_keywords(self):
        if os.path.exists(KEYWORDS_FILE):
            with open(KEYWORDS_FILE, "r") as f:
                return json.load(f)
        else:
            print(f"[ToxiGuard] {KEYWORDS_FILE} not found. Using default keywords.")
            return {
                "slurs": {"words": ["faggot", "nigga", "coon"], "severity": 5},
                "insults": {"words": ["stupid", "idiot", "dumb", "trash", "noob"], "severity": 2},
                "violence": {"words": ["kill yourself", "die", "kys"], "severity": 5},
                "swearing": {"words": ["fuck", "bitch"], "severity": 3}
            }

    def transcribe_audio(self, filepath):
        print("[ToxiGuard] Transcribing audio...")
        result = self.model.transcribe(filepath)
        print(f"[ToxiGuard] Transcription result: {result['text']}")
        return result['text']

    def check_toxicity(self, text):
        flagged = []
        for category, data in self.toxic_keywords.items():
            if data["severity"] >= self.severity_threshold:
                for word in data["words"]:
                    if word.lower() in text.lower():
                        flagged.append({
                            "word": word,
                            "category": category,
                            "severity": data["severity"]
                        })
        return flagged

    def transcribe_and_score(self, transcription, filepath):
        transcription_path = os.path.join(self.recorder.output_dir, "transcription.txt")
        report_path = os.path.join(self.recorder.output_dir, "toxicity_report.json")

        with open(transcription_path, "w") as f:
            f.write(transcription)

        found_keywords = self.check_toxicity(transcription)
        if found_keywords:
            total_severity = sum(word["severity"] for word in found_keywords)
            max_possible = sum(data["severity"] * len(data["words"]) for data in self.toxic_keywords.values())
            score = total_severity / max_possible
        else:
            score = 0.0

        report = {
            "transcription": transcription,
            "toxicity_score": score,
            "flagged_words": found_keywords
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[ToxiGuard] Transcription saved to: {transcription_path}")
        print(f"[ToxiGuard] Toxicity report saved to: {report_path}")

    def run_monitor_loop(self, interval=5):
        print("[ToxiGuard] Starting background monitoring...")
        print(f"[ToxiGuard] Current Working Directory: {os.getcwd()}")
        while True:
            print("[ToxiGuard] Listening...")
            filepath = self.recorder.record_system_audio()
            transcription = self.transcribe_audio(filepath)
            print(f"[ToxiGuard] Transcribed Text: {transcription}")
            found = self.check_toxicity(transcription)
            print(f"[ToxiGuard] Found toxic words: {found}")
            if found:
                print("[ToxiGuard] Toxic behavior detected! Saving clip and report...")
                self.transcribe_and_score(transcription, filepath)
            print("[ToxiGuard] Loop complete. Waiting for next check...")
            time.sleep(interval)

if __name__ == '__main__':
    app = ToxiGuardBackend()
    app.run_monitor_loop()
