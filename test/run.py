from query import AudioRecorder
from chat_rag import ChatSystem
from tts import TextToSpeechGenerator
from stt import AudioTranscriber



chat_system = ChatSystem()#LLM with RAG
transcriber = AudioTranscriber() #STT
recorder = AudioRecorder(filename="question.wav")#input with microphone
tts_generator = TextToSpeechGenerator()#TTS
log_file_path = "./log_p-guard1.json"
chat_system.mount(log_file_path)
while True:
    recorder.keyboard_listener()
    transcription_text = transcriber.transcribe("question.wav")
    answer = chat_system.run(transcription_text)
    print(answer)
    output_file = tts_generator.generate_speech(answer)
    