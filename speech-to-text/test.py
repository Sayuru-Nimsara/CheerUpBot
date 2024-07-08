import pyaudio
import wave
import os
from faster_whisper import WhisperModel

format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
chunkLen = 1
file_path = "recording.wav"

def recordAudio(file_path, audio, stream):
    frames = []
    for _ in range(0,int(rate / chunk * chunkLen)):
        data = stream.read(chunk)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format)) 
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # print("press space to Recoding...")
    # keyboard.wait('space')
    # print("recording... press space to stop")
    # time.sleep(0.2)
     
    # while True:
    #     try:
    #         data = stream.read(chunk)
    #         frames.append(data) 
    #     except KeyboardInterrupt:
    #         break
    #     if keyboard.is_pressed('space'):
    #         print("Stop recording...")
    #         time.sleep(0.2)  
    #         break
    
    # stream.stop_stream()
    # stream.close()
    # audio.terminate()
    
def main2():
    modelSize = "medium.en"
    model = WhisperModel(modelSize, device="cuda", compute_type="float16")
    audio = pyaudio.PyAudio()
    stream = audio.open(format = format, channels =channels, rate= rate, input= True, frames_per_buffer = chunk)
    accumulate_transcription = ""
    
    try:
        while True:
            
            recordAudio(file_path, audio, stream)
            transcription = model.transcribe(file_path)
            print(transcription)
            os.remove(file_path)
            
            accumulate_transcription += transcription + ""
    except KeyboardInterrupt:
        print("Stopping..")
        with open("lg.txt", "w") as log_file:
            log_file.write(accumulate_transcription)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
if __name__ == "__main__":
    main2()

# recordAudio("my.wav")  