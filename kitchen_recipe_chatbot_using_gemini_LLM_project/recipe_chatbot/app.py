import os
import re
import time

from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load environment variables
load_dotenv()

# Configure the GenAI API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key is not set. Please check your .env file.")
genai.configure(api_key=api_key)

# Initialize the generative model and chat history
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Initialize Streamlit app
st.set_page_config(page_title="Kitchen Recipe Chatbot", layout="centered")
st.header("🍲 Kitchen Recipe Chatbot")

# Define the initial prompt format
input_prompt = """
You are a culinary expert and kitchen assistant, helping users find recipes based on available ingredients.
When a user provides a list of ingredients or asks about a specific dish, respond with a recipe that includes:

1. **Recipe Name** : (Give the name of the recipe with Normal font)
2. **Ingredients**: (list with quantities)
3. **Instructions**: (step-by-step cooking guide)
4. **Additional Tips**: (optional, for preparation or substitutions)

Format the response accordingly and be concise yet thorough.
"""

# Initialize session state for chat history and responses if not set
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "bot_responses" not in st.session_state:
    st.session_state["bot_responses"] = []


# Function to preprocess user input text
def preprocess_text(text):
    # Remove all the special characters
    text = re.sub(r"\W", " ", str(text))

    # Remove all single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)

    # Remove single characters from the start
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)

    # Substitute multiple spaces with a single space
    text = re.sub(r"\s+", " ", text, flags=re.I)

    # Remove prefixed 'b'(byte strings)
    text = re.sub(r"^b\s+", "", text)

    # Convert to lowercase
    text = text.lower()

    return text.strip()


# Defining a function to remove stopwords from the review
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))

    # Split the text into words
    words = text.split()

    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]

    # Join the words back into a single string
    return " ".join(filtered_words)


# Defining a function to lemmatize the input text
def lemmatize_input_text(text):
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text into words
    words = [word for word in text.split()]

    # Lemmatize each word
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)


# preprocessing function to clean the input text
def clean_input_text(text):
    preprocessed_text = preprocess_text(text)
    text_without_stopwords = remove_stopwords(preprocessed_text)
    cleaned_text = lemmatize_input_text(text_without_stopwords)
    return cleaned_text


# Text area input field
user_input = st.text_area(
    "Type your recipe request here:",
    placeholder="Enter ingredients or dish name...",
    key="input",
    height=100,
)
submit_button = st.button("Send")

# Display chat history below the input box
for role, text in st.session_state["chat_history"]:
    if role == "You":
        st.write(f"**{role}:** {text}")
    else:
        st.write(f"**Bot:** {text}")

# Divider for layout organization
st.divider()


# Function to handle retries
def get_bot_response_with_retry(user_input, retries=5, delay=5):
    for attempt in range(retries):
        try:
            # Send prompt + query to model in a single API call
            response_chunks = chat.send_message(
                f"{input_prompt}\n\nUser query: {user_input}", stream=True
            )

            # Collect the response text
            bot_reply = "".join([chunk.text for chunk in response_chunks])
            return bot_reply

        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Error: {e}. Retrying in {delay} seconds...")

                # Wait before retrying
                time.sleep(delay)
            else:
                st.error(f"Error: {e}. Please try again later.")
                return None


# Extract the recipe name for the filename
def extract_recipe_name(recipe_text):
    # Matching various patterns for a Recipe Name in the response
    match = re.search(r"(Recipe Name[:\-]?\s*)([^\n]+)", recipe_text, re.IGNORECASE)
    if match:
        recipe_name = match.group(1).strip()

        # Clean up the recipe name for use as a filename
        recipe_name = re.sub(r'[\\/*?:"<>|]', "", recipe_name)
        return recipe_name
    else:
        # Log output
        print("Recipe name not found. Using default filename.")
    return "recipe"


# Process user input and get response on form submission
if submit_button and user_input:
    # Clean user input
    cleaned_input = clean_input_text(user_input)

    # Add original user query to chat history
    st.session_state["chat_history"].append(("You", user_input))

    # Get response from Gemini model with retry mechanism using cleaned input
    bot_reply = get_bot_response_with_retry(cleaned_input)

    if bot_reply:
        # Update chat history and bot responses
        st.session_state["chat_history"].append(("Bot", bot_reply))
        # Track bot responses for saving
        st.session_state["bot_responses"].append(bot_reply)

        # Immediately display the bot response after processing
        st.write(f"**Bot:** {bot_reply}")

# Save Button to download recipe as a text file
if st.session_state["bot_responses"]:
    recipe = st.session_state["bot_responses"][-1]
    recipe_name = extract_recipe_name(recipe)
    if " " in recipe_name:
        recipe_name = recipe_name.lstrip().replace(" ", "_") + "_recipe"
    else:
        recipe_name = recipe_name.lstrip() + "_recipe"
    st.download_button(
        label="Save Recipe",
        data=recipe,
        file_name=f"{recipe_name}.txt",
        mime="text/plain",
    )

# Display prompt for first-time users if chat history is empty
if not st.session_state["chat_history"]:
    st.info("Enter a recipe query to start chatting with the bot.")
