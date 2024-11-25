from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

class TextToSpeechGenerator:
    def __init__(self, model_name="speecht5_tts", vocoder_name="speecht5_hifigan"):
        # Load processor, model, and vocoder
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)

        # Load a sample xvector for speaker embeddings
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    def generate_speech(self, text, speaker_index=7306, output_path="speech.wav"):
        # Process the input text
        inputs = self.processor(text=text, return_tensors="pt")

        # Get speaker embeddings from dataset based on index
        speaker_embeddings = torch.tensor(self.embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)

        # Generate speech
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)

        # Save to file
        sf.write(output_path, speech.numpy(), samplerate=16000)

        return output_path

# Usage
#tts_generator = TextToSpeechGenerator()
#output_file = tts_generator.generate_speech(
#    text="""Dancing in the masquerade, idle truth and plain sight jaded, pop, roll, click, shot, 
#    who will I be today or not? But such a tide as moving seems asleep, too full for sound and foam, 
#    when that witch drew from out the boundless deep turns again home, twilight and evening bell and after that""")
#print(f"Audio generated and saved to {output_file}")
