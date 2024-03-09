# YT video: https://www.youtube.com/watch?v=_j7JEDWuqLE

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    #text = image_to_text(url) # [{'generated_text': 'a man with a beard and a blue jacket'}]
    text = image_to_text(url)[0]['generated_text']
    return text

# LLM
def create_story(text):
    #text_to_story = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
    #story = text2speach
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    # def query(payload):
    #     response = requests.post(API_URL, headers=headers, json=payload)
    #     return response.json()
    
    input = f'Create a short 50 word story using this as an input: "{text}"'
    payloads = {
        "inputs": input,
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    return response.json()[0]['generated_text'].split('"')[2]

# text to speech
def text2speach(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {
        'inputs': message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="Image 2 Audio Story", page_icon='🍌')
    st.header('Turn image into audio story')
    uploaded_file = st.file_uploader('Choose an image', type='jpg')

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        text = img2text(uploaded_file.name)
        story = create_story(text)
        text2speach(story)

        with st.expander('Story'):
            st.write(story)
        st.audio('audio.flac')

if __name__ == '__main__':
    main()