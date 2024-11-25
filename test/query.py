import sounddevice as sd
import numpy as np
import threading
import keyboard
from scipy.io.wavfile import write

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.3, silence_threshold=200, mic_index=0, filename="question.wav"):
        # Set parameters
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.mic_index = mic_index
        self.filename = filename
        
        # Initialize recording variables
        self.recording = False
        self.audio_data = []

    def start_recording(self):
        """Capture audio in chunks and stop on silence."""
        self.audio_data = []  # Reset audio data buffer
        self.recording = True
        print("Recording started...")

        # Continuously capture audio in chunks
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', device=self.mic_index) as stream:
            silent_chunks = 0  # Track consecutive silent chunks
            while self.recording:
                chunk, overflowed = stream.read(int(self.sample_rate * self.chunk_duration))
                self.audio_data.append(chunk)

                # Check if current chunk is silent
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # Reset count if there's sound

                # Stop recording if silence persists for a few chunks
                if silent_chunks > 3:  # Adjust this to control silence duration
                    self.recording = False
                    print("Silence detected. Stopping recording...")

    def start_recording_thread(self):
        """Start recording in a separate thread and save audio."""
        record_thread = threading.Thread(target=self.start_recording)
        record_thread.start()
        record_thread.join()  # Wait for recording thread to complete
        
        # Convert audio data to a single numpy array and save as .wav
        audio_np = np.concatenate(self.audio_data, axis=0)
        write(self.filename, self.sample_rate, audio_np)
        print(f"Audio saved to '{self.filename}'")

    def keyboard_listener(self):
        """Listen for spacebar press to start recording."""
        print("Press 'space' to start recording.")
        keyboard.wait('space')  # Wait for the spacebar press to start recording

        if not self.recording:
            self.start_recording_thread()  # Start recording

# Usage
#recorder = AudioRecorder(filename="question.wav")
#recorder.keyboard_listener()
