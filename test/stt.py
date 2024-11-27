from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torchaudio.transforms as T
import torch

class AudioTranscriber:
    def __init__(self, model_name="whisper-base"):
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def open_audio(self, file_path):
        # Load audio file
        signal, sampling_rate = torchaudio.load(file_path)

        # Convert stereo to mono if needed
        if signal.size(0) > 1:  # More than 1 channel
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Resample to 16 kHz if needed
        target_sampling_rate = 16000
        if sampling_rate != target_sampling_rate:
            resampler = T.Resample(sampling_rate, target_sampling_rate)
            signal = resampler(signal)
            sampling_rate = target_sampling_rate

        return signal, sampling_rate

    def transcribe(self, file_path):
        # Load and preprocess audio
        signal, sampling_rate = self.open_audio(file_path)

        # Ensure signal has the correct shape for batch processing
        if len(signal.shape) == 2 and signal.shape[0] == 1:  # (1, num_samples)
            signal = signal.squeeze(0)  # Remove channel dimension

        # Process input features
        input_features = self.processor(signal, sampling_rate=sampling_rate, return_tensors="pt").input_features

        # Generate predicted IDs and decode to text
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription[0]

# Usage
#transcriber = AudioTranscriber()
#transcription_text = transcriber.transcribe("speech.wav")  # Replace with your audio file path
#print("Transcription:", transcription_text)
