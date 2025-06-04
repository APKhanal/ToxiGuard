import soundcard as sc
import numpy as np
import wave
import os
import sys
import threading
import subprocess
import time
import whisper
import json

TOXIC_KEYWORDS = [
    "kill yourself", "retard", "trash", "noob", "stupid", "idiot",
    "dumb", "f***", "b****", "n****", "c****", "kys", "die"
]

class AudioRecorder:
    def __init__(self, sample_rate=44100, buffer_seconds=15):
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ToxiGuard_Output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.loopback_mic = self._get_loopback_microphone()

    def _get_loopback_microphone(self):
        print("[ToxiGuard] Searching for VB-Cable device (CABLE Output)...")
        all_mics = sc.all_microphones()
        for mic in all_mics:
            print(f" - {mic.name}")
        mic = next((m for m in all_mics if 'cable output' in m.name.lower()), None)
        if mic is None:
            raise RuntimeError("[ToxiGuard] Could not find 'CABLE Output'. Make sure it's enabled and active.")
        print(f"[ToxiGuard] Using loopback mic: {mic.name}")
        return mic

    def start_recording(self):
        print(f"[ToxiGuard] Monitoring system audio via: {self.loopback_mic.name}")
        print(f"[ToxiGuard] Output files will be saved to: {self.output_dir}")

    def get_last_seconds(self, seconds):
        numframes = int(self.sample_rate * seconds)
        with self.loopback_mic.recorder(samplerate=self.sample_rate) as rec:
            data = rec.record(numframes=numframes)
        return data

    def save_to_wav(self, data, filename):
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        filepath = os.path.join(self.output_dir, filename)
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
        with self.recorder.loopback_mic.recorder(samplerate=self.recorder.sample_rate) as rec:
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
        ##os.remove(tmp_path)##
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
