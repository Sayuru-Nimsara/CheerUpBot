import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import re

# Initialize speech recognizer, text-to-speech engine
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 400

engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Change voice (if multiple voices are available)
engine.setProperty('voice', voices[1].id)  # Change index to select a different voice

# Adjust other parameters (optional)
engine.setProperty('rate', 170)  # Speed
engine.setProperty('volume', 0.9)  # Volume


# Set Google Gemini API key as a system environment variable or add it here
genai.configure(api_key="AIzaSyDgQIZfu1h4OZJa4v1cNHBe44xsQ3pG33c")

# Model of Google Gemini API
model = genai.GenerativeModel('gemini-1.5-pro',
    generation_config=genai.GenerationConfig(
        candidate_count=1,
        top_p=0.5,
        top_k=5,
        max_output_tokens=150,  # Increased to 150 tokens
        temperature=0.7,
    ))

# Start the chat model 
chat = model.start_chat(history=[])

def remove_emojis(text):
    # This regex matches emojis
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "\*"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)



# Function to convert speech to text
def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("recording...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
            text = recognizer.recognize_google(audio)
            print("User: ", text)
            return text
        except sr.UnknownValueError:
            return "couldn't understand"
        except sr.RequestError as e:
            return "Could not request results; {0}".format(e)

# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def generate_response(text):
    chat.send_message(text)
    clean_text = remove_emojis(chat.last.text)
    print("Response: "+ clean_text)
    return clean_text


def process_text(text):
    response = generate_response(text)  
    return response

# Main loop
isSpeaking  = True
while isSpeaking:
    user_input = speech_to_text()
    if(user_input=="thank you"):
        text_to_speech("No problem at all! I'm always here if you need me. Just tell anytime.")
        isSpeaking = False
    else:
        response = process_text(user_input)
        text_to_speech(response)
    
