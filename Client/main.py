import requests
import pyaudio
import wave
import os
import time
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

# Server API URL
SERVER_URL = "http://10.7.0.28:5508/process_audio"

# Paths to audio feedback files
START_LISTENING_SOUND = "start_listening_sound.mp3"
STOPPED_LISTENING_SOUND = "stopped_listening_sound.mp3"

# Function to play sound using pydub
def play_sound(sound_file):
    sound = AudioSegment.from_mp3(sound_file)  # Load the MP3 file
    play(sound)  # Play the sound

def record_audio():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        # Adjust for ambient noise
        # recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust to ambient noise for 1 second
        
        # # Optionally, adjust the energy threshold manually (based on test)
        # recognizer.energy_threshold = 4000  # Adjust this value based on your environment
        
        play_sound(START_LISTENING_SOUND)

        print("Listening...")
        try:
            # Capture audio from microphone
            audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
            
            # If we get here, speech was detected, so process it
            play_sound(STOPPED_LISTENING_SOUND)

            # Save the recorded audio as a WAV file
            audio_file_path = "input_audio.wav"
            with open(audio_file_path, 'wb') as f:
                f.write(audio.get_wav_data())
                
            # print("Audio recorded successfully.")
            return audio_file_path
        
        except sr.WaitTimeoutError:

            play_sound(STOPPED_LISTENING_SOUND)
            print("No speech detected, please try again.")
            return None

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

# Send the recorded audio to the server and receive response with retry on failure
def send_to_server(audio_file_path):
    max_retries = 3
    for attempt in range(max_retries):
        print("Sending audio to server... (Attempt {}/{})".format(attempt + 1, max_retries))
        try:
            with open(audio_file_path, 'rb') as f:
                response = requests.post(SERVER_URL, files={'audio_file': f})
            if response.status_code == 200:
                response_audio_path = "response_audio.mp3"
                with open(response_audio_path, 'wb') as audio_file:
                    audio_file.write(response.content)
                print("Response received.")
                return response_audio_path
            else:
                print(f"Server error: {response.status_code} - {response.text}")
                return None
        except requests.ConnectionError:
            print(f"Connection error. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(1)
    print("Failed to connect to the server after multiple attempts.")
    return None

# Play the response audio directly
def play_audio(response_audio_path):
    if response_audio_path:
        print("Playing server response...")
        audio = AudioSegment.from_file(response_audio_path)
        play(audio)
    else:
        print("No audio to play.")

if __name__ == "__main__":
    while True:
        # Step 1: Record audio with silence detection
        audio_file_path = record_audio()
        if not audio_file_path:
            continue  # Restart loop if no audio was captured

        # Step 2: Send recorded audio to the server and get response
        print("Processing your request, please wait...")
        response_audio_path = send_to_server(audio_file_path)

        # Step 3: Play the received audio response
        play_audio(response_audio_path)



