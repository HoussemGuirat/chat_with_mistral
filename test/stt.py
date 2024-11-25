from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio2numpy import open_audio

class AudioTranscriber:
    def __init__(self, model_name="whisper-base"):
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def open_audio(self, file_path):
        # Function to load audio file and return signal and sampling rate
        import torchaudio
        signal, sampling_rate = torchaudio.load(file_path)
        return signal, sampling_rate

    def transcribe(self, file_path):
        # Load audio
        signal, sampling_rate = self.open_audio(file_path)

        # Process input features
        input_features = self.processor(signal, sampling_rate=sampling_rate, return_tensors="pt").input_features

        # Generate predicted IDs and decode to text
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription[0]

#transcriber = AudioTranscriber()
#transcription_text = transcriber.transcribe("question.wav")  # Update with the correct file path
#print(transcription_text)
