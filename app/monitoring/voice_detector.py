import webrtcvad
import pyaudio
import numpy as np
from typing import Optional
import threading
import time


class VoiceDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode: 0-3, 3 is most aggressive
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self.speech_detected = False
        self.last_speech_time: Optional[float] = None
        self.speech_duration = 0.0

    def start_monitoring(self):
        """Start monitoring microphone for voice/speech."""
        if self.is_monitoring:
            return

        try:
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size,
            )
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_voice, daemon=True)
            self.monitoring_thread.start()
        except Exception as e:
            print(f"Error starting voice monitoring: {e}")
            # If microphone is not available, continue without voice detection
            self.is_monitoring = False

    def _monitor_voice(self):
        """Monitor voice in background thread."""
        speech_start_time: Optional[float] = None
        
        while self.is_monitoring:
            try:
                if self.stream is None:
                    break
                    
                frame = self.stream.read(self.frame_size, exception_on_overflow=False)
                
                # Check if frame contains speech using VAD
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                
                if is_speech:
                    if speech_start_time is None:
                        speech_start_time = time.time()
                    else:
                        # Check if speech has been continuous for a threshold
                        speech_duration = time.time() - speech_start_time
                        # If speech detected for more than 0.5 seconds, consider it human speech
                        if speech_duration > 0.5:
                            self.speech_detected = True
                            self.speech_duration = speech_duration
                else:
                    # Reset if no speech detected
                    if speech_start_time is not None:
                        # Brief pauses are okay, reset after 0.3 seconds
                        if time.time() - speech_start_time - self.speech_duration > 0.3:
                            speech_start_time = None
                            self.speech_detected = False
                            self.speech_duration = 0.0

            except Exception as e:
                print(f"Error in voice monitoring: {e}")
                break

    def is_human_speech_detected(self) -> bool:
        """
        Check if human speech is detected.
        Returns True if speech has been detected continuously.
        """
        return self.speech_detected

    def process_audio_frame(self, audio_data: bytes) -> bool:
        """
        Process audio frame from external source (e.g., from frontend).
        Returns True if speech is detected.
        """
        try:
            if len(audio_data) < self.frame_size * 2:  # 2 bytes per sample
                return False
                
            # Check if frame contains speech using VAD
            is_speech = self.vad.is_speech(audio_data[:self.frame_size * 2], self.sample_rate)
            return is_speech
        except Exception as e:
            print(f"Error processing audio frame: {e}")
            return False

    def stop_monitoring(self):
        """Stop monitoring microphone."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.speech_detected = False
        self.last_speech_time = None
        self.speech_duration = 0.0

    def release(self):
        """Release resources."""
        self.stop_monitoring()
        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None

