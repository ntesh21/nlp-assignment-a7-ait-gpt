from flask import Flask, render_template, request, jsonify
import random
from chatbot import chatbot

app = Flask(__name__)


# Function to generate a random response from the list of predefined responses
def generate_response(query):
    response = chatbot(query)
    return response

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle user input and return the chatbot response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    bot_response = generate_response(user_message)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
