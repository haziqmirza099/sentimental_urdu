"""import streamlit as st
import speech_recognition as sr
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from pydub import AudioSegment
from urduhack.models.lemmatizer import lemmatizer
from p import UrduTextPreprocessor


cv_loaded = load('countvectorizer_updateddr4.joblib')
model = load('sentiNB_updateddr4.joblib')
preprocessor = UrduTextPreprocessor()

def main():
    st.title("Urdu Sentiment Analysis")
    option = st.selectbox("Select mode:", ("File-based Audio Processing", "Real-time Audio Processing"))

    if option == "File-based Audio Processing":
        audio_file = st.file_uploader("Upload audio file (WAV format):", type=["wav"])
        if st.button("Analyze Audio") and audio_file:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            preprocessed_text = preprocessor.recognize_and_preprocess_audio("temp_audio.wav")
            if preprocessed_text:
                st.write("Preprocessed Transcription:", preprocessed_text)
                if preprocessor.detect_abusive_words(preprocessed_text):
                    st.error("Abusive Sentence 🚫")
                else:
                    X_sentence = cv_loaded.transform([preprocessed_text])
                    prediction = model.predict(X_sentence)
                    if prediction[0] == 1:
                        st.error("Negative Sentence 😠")
                    elif prediction[0] == 0:
                        st.success("Positive Sentence 😊")
                    else:
                        st.warning("Neutral Sentence 😐")

    elif option == "Real-time Audio Processing":
        st.info("Click on the 'Start Recording' button to begin recording audio from your microphone.")

        
        recognizer = sr.Recognizer()

        
        def start_recording():
            st.info("Recording... Say something!")
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=None)
                process_audio(recognizer, audio)
            except Exception as e:
                st.error(f"Error while recording: {e}")

        
        def stop_recording():
            st.info("Stopped recording.")

        
        def process_audio(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio, language="ur-PK")
                preprocessed_text = preprocessor.preprocess(text)
                st.write("Preprocessed Transcription:", preprocessed_text)

                if preprocessed_text:
                    if preprocessor.detect_abusive_words(preprocessed_text):
                        st.error("Abusive Sentence 🚫")
                    else:
                        X_sentence = cv_loaded.transform([preprocessed_text])
                        prediction = model.predict(X_sentence)

                        if prediction[0] == 1:
                            st.error("Negative Sentence 😠")
                        elif prediction[0] == 0:
                            st.success("Positive Sentence 😊")
                        else:
                            st.warning("Neutral Sentence 😐")

            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                st.error(f"Error processing audio: {e}")

        # UI buttons to start and stop recording
        if st.button("Start Recording"):
            start_recording()

        if st.button("Stop Recording"):
            stop_recording()

if __name__ == "__main__":
    main()
"""
import streamlit as st
import speech_recognition as sr
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from urduhack.models.lemmatizer import lemmatizer
from p import UrduTextPreprocessor
from st_audiorec import st_audiorec

# Load pre-trained models and preprocessor
cv_loaded = load('countvectorizer_updateddr4.joblib')
model = load('sentiNB_updateddr4.joblib')
preprocessor = UrduTextPreprocessor()

def process_audio_file(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="ur-PK")
            preprocessed_text = preprocessor.preprocess(text)
            st.write("Preprocessed Transcription:", preprocessed_text)

            if preprocessed_text:
                if preprocessor.detect_abusive_words(preprocessed_text):
                    st.error("Abusive Sentence 🚫")
                else:
                    X_sentence = cv_loaded.transform([preprocessed_text])
                    prediction = model.predict(X_sentence)

                    if prediction[0] == 1:
                        st.error("Negative Sentence 😠")
                    elif prediction[0] == 0:
                        st.success("Positive Sentence 😊")
                    else:
                        st.warning("Neutral Sentence 😐")

        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            st.error(f"Error processing audio: {e}")

def main():
    st.title("Urdu Sentiment Analysis")
    option = st.selectbox("Select mode:", ("File-based Audio Processing", "Real-time Audio Processing"))

    if option == "File-based Audio Processing":
        audio_file = st.file_uploader("Upload audio file (WAV format):", type=["wav"])
        if st.button("Analyze Audio") and audio_file:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            preprocessed_text = preprocessor.recognize_and_preprocess_audio("temp_audio.wav")
            if preprocessed_text:
                st.write("Preprocessed Transcription:", preprocessed_text)
                if preprocessor.detect_abusive_words(preprocessed_text):
                    st.error("Abusive Sentence 🚫")
                else:
                    X_sentence = cv_loaded.transform([preprocessed_text])
                    prediction = model.predict(X_sentence)
                    if prediction[0] == 1:
                        st.error("Negative Sentence 😠")
                    elif prediction[0] == 0:
                        st.success("Positive Sentence 😊")
                    else:
                        st.warning("Neutral Sentence 😐")

    elif option == "Real-time Audio Processing":
        st.info("Click on the 'Start Recording' button to begin recording audio from your microphone.")

        
        wav_audio_data = st_audiorec()

        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')

            
            with open("temp_realtime_audio.wav", "wb") as f:
                f.write(wav_audio_data)  

            
            process_audio_file("temp_realtime_audio.wav")

if __name__ == "__main__":
    main()
