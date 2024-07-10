import openai
import os
import json
import constants
from flask import Flask, request, jsonify
import random
import time

app = Flask(__name__)

# Set the API key
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(user_input):
    # Example predefined responses
    predefined_responses = {
        "hello": "Hi there! How can I help you today?",
        "how are you": "I'm just a bot, but I'm here to assist you!",
        "what is your name": "I am ChatGPT, your virtual assistant.",
        "bye": "Goodbye! Have a great day!"
    }
    
    # Check if the user input matches any predefined responses
    for key in predefined_responses:
        if key in user_input:
            return predefined_responses[key]
    
    # If no match is found, return "not found"
    return "not found"

def format_links(response):
    # Placeholder function to format links
    # Replace with actual implementation if needed
    return response

@app.route('/chat', methods=['POST'])
def get_chatgpt_response():
    try:
        user_input = request.json["user_input"].lower()  # get the user input
        
        response = get_response(user_input)  # get response from custom data
        if any(substring in response for substring in ['http://', 'https://', 'www.']):
            response = format_links(response)
        
        if response == "not found":
            prompt = f"User: {user_input}\n\nChatGPT:"  # Prompt for GPT-3.5-Turbo, ensuring polite and helpful responses
            
            # Call OpenAI API with appropriate parameters
            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,  # Adjust response length as needed
                n=1,
                stop=None,
                temperature=0.7,  # Adjust creativity as needed
            )
            
            response = openai_response['choices'][0]['message']['content']
            
        delay_seconds = random.randint(3, 5)
        time.sleep(delay_seconds)
        return jsonify({'chatgpt_response': response}) 
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
