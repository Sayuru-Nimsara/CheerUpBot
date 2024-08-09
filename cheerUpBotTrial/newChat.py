import google.generativeai as genai
import speech_recognition as sr
from datetime import date
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import threading
import queue
import time
import re

# Initialize mixer
mixer.init()
mixer.set_num_channels(1)
voice = mixer.Channel(0)

# Set Google Gemini API key as a system environment variable or add it here
genai.configure(api_key="AIzaSyDu22YpdWHMHIi2Jg7hNDSmBy6kKa2NNrs")

# Initialize Google Gemini API model
model = genai.GenerativeModel(
    'gemini-1.5-pro',
    generation_config=genai.GenerationConfig(
        candidate_count=1,
        top_p=0.7,
        top_k=4,
        max_output_tokens=500,  # 100 tokens correspond to roughly 60-80 words.
        temperature=0.7,
    )
)

# Start chat model
chat = model.start_chat(history=[])

today = str(date.today())

# Initialize counters
numtext = 0
numtts = 0
numaudio = 0


# Function to remove emojis and special characters
def clean_text(text):
    # Remove emojis and special characters
    clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    clean_text = re.sub(r'[^A-Za-z0-9\s,.!?\'"]+', '', clean_text)
    return clean_text


# Thread for text generation
def chatfun(request, text_queue, llm_done):
    global numtext, chat

    try:
        response = chat.send_message(request, stream=True)
        print("Received response from Google Gemini API")

        for chunk in response:
            if chunk.candidates[0].content.parts:
                raw_text = chunk.candidates[0].content.parts[0].text
                clean_response = clean_text(raw_text)
                print(clean_response, end='')
                text_queue.put(clean_response.replace("*", ""))
                time.sleep(0.2)
                numtext += 1

        append2log(f"AI: {clean_response}\n")
        llm_done.set()
    except Exception as e:
        print(f"Error in chatfun: {e}")
        llm_done.set()


# Convert text to speech and play it
def speak_text(text):
    try:
        mp3file = BytesIO()
        tts = gTTS(text, lang="en", tld='us')
        tts.write_to_fp(mp3file)
        mp3file.seek(0)

        sound1 = mixer.Sound(mp3file)
        voice.play(sound1)

        print("AI:", text)

        while voice.get_busy():
            time.sleep(0.01)

        mp3file = None
    except Exception as e:
        print(f"Error in speak_text: {e}")


# Thread for text-to-speech conversion
def text2speech(text_queue, tts_done, llm_done, audio_queue, stop_event):
    global numtext, numtts

    time.sleep(1.0)

    while not stop_event.is_set():
        try:
            if not text_queue.empty():
                text = text_queue.get(timeout=0.5)

                if len(text) > 1:
                    mp3file1 = BytesIO()
                    tts = gTTS(text, lang="en", tld='us')
                    tts.write_to_fp(mp3file1)

                    audio_queue.put(mp3file1)
                    numtts += 1
                    text_queue.task_done()
                else:
                    print("Skipping text:", text)
                    text_queue.task_done()

            if llm_done.is_set() and numtts == numtext:
                time.sleep(0.2)
                tts_done.set()
                break
        except Exception as e:
            print(f"Error in text2speech: {e}")
            tts_done.set()


# Thread for audio playback
def play_audio(audio_queue, tts_done, stop_event):
    global numtts, numaudio

    while not stop_event.is_set():
        try:
            mp3audio1 = audio_queue.get()
            mp3audio1.seek(0)
            sound1 = mixer.Sound(mp3audio1)
            voice.play(sound1)

            numaudio += 1
            audio_queue.task_done()

            while voice.get_busy():
                time.sleep(0.01)

            if tts_done.is_set() and numtts == numaudio:
                break
        except Exception as e:
            print(f"Error in play_audio: {e}")
            tts_done.set()
            break


# Save conversation to a log file
def append2log(text):
    global today
    fname = 'chatlog-' + today + '.txt'
    with open(fname, "a", encoding='utf-8') as f:
        f.write(text + "\n")


# Function to start a conversation with hardcoded responses
def start_conversation():
    initial_responses = [
        "Hey, how’s it going? You seem a bit bored. Everything okay? Feel free to ask me anything",
    ]

    for response in initial_responses:
        append2log(f"AI: {response}\n")
        speak_text(response)
        time.sleep(1)  # Pause between responses


# Main function
def main():
    global today, chat, model, numtext, numtts, numaudio

    # Start the conversation with hardcoded responses
    start_conversation()
    isConversationGoing = True
    rec = sr.Recognizer()
    mic = sr.Microphone()
    rec.dynamic_energy_threshold = False
    rec.energy_threshold = 400

    while isConversationGoing:
        with mic as source:
            rec.adjust_for_ambient_noise(source, duration=1)

            print("Listening ...")

            try:
                audio = rec.listen(source, timeout=20, phrase_time_limit=30)
                print("Audio captured")
                text = rec.recognize_google(audio, language="en")
                print(f"Recognized text: {text}")

                request = text.lower()

                if "that's all" in request:
                    append2log(f"You: {request}\n")
                    speak_text("Bye now")
                    append2log("AI: Bye now.\n")
                    isConversationGoing = False
                    continue

                if "jack" in request:
                    request = request.split("jack")[1]

                # Process user's request
                append2log(f"You: {request}\n")
                print(f"You: {request}\nAI:", end='')

                # Initialize the counters before each reply from AI
                numtext = 0
                numtts = 0
                numaudio = 0

                # Define text and audio queues for data storage
                text_queue = queue.Queue()
                audio_queue = queue.Queue()

                # Define events
                llm_done = threading.Event()
                tts_done = threading.Event()
                stop_event = threading.Event()

                # Thread for handling the LLM responses
                llm_thread = threading.Thread(target=chatfun, args=(request, text_queue, llm_done,))
                llm_thread.start()

                # Thread for text-to-speech
                tts_thread = threading.Thread(target=text2speech,
                                              args=(text_queue, tts_done, llm_done, audio_queue, stop_event,))
                tts_thread.start()

                # Thread for audio playback
                play_thread = threading.Thread(target=play_audio, args=(audio_queue, tts_done, stop_event,))
                play_thread.start()

                # Wait for LLM to finish responding
                llm_done.wait()
                llm_thread.join()

                tts_done.wait()
                audio_queue.join()

                stop_event.set()
                tts_thread.join()
                play_thread.join()

                print('\n')

            except Exception as e:
                print(f"An error occurred: {e}")
                continue


if __name__ == "__main__":
    main()
