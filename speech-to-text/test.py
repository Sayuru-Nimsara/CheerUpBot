import pyaudio
import wave
import os
import keyboard
from faster_whisper import WhisperModel
import time
  

format = pyaudio.paInt16
channels = 1
rate = 46000
chunk = 1024
chunkLen = 1
file_path = "recording.wav"

def recordAudio(file_path, audio, stream):
    frames = []
    for _ in range(0,50): 
        data = stream.read(chunk)
        frames.append(data)
    # data = stream.read(chunk)
    # frames.append(data) 
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format)) 
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
     
    # print("press space to Recoding...")
    # keyboard.wait('space')
    # print("recording... press space to stop")
     
    # while True:
    #     try:
    #         data = stream.read(chunk)
    #         frames.append(data)  
    #     if keyboard.is_pressed('space'):
    #         print("Stop recording...")
    #         time.sleep(0.2)  
    #         break
    # wf = wave.open(file_path, 'wb')
    # wf.setnchannels(channels)
    # wf.setsampwidth(audio.get_sample_size(format)) 
    # wf.setframerate(rate)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    # stream.stop_stream()
    # stream.close()
    # audio.terminate()
    
def main2():
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="int8") 
  
    audio = pyaudio.PyAudio()
    stream = audio.open(format = format, channels =channels, rate= rate, input= True, frames_per_buffer = chunk)
    accumulate_transcription = ""
    
    try:
        print("Press space")
        keyboard.wait('space')
        while True:
            recordAudio(file_path, audio, stream)
            segments, _ = model.transcribe(file_path,beam_size=5)
              
            for segment in segments:
                transcription = segment.text
                # print("%s" % (segment.text))
                print(transcription)
                accumulate_transcription += segment.text + " "
            
            os.remove(file_path)
            
            print("recoding again...")
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