import streamlit as st
import spacy
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
from urduhack.preprocessing import (
    normalize_whitespace,
    remove_punctuation,
    remove_accents,
    replace_urls,
    replace_emails,
    replace_numbers,
    replace_currency_symbols,
)
from urduhack.models.lemmatizer import lemmatizer

class UrduTextPreprocessor:
    def __init__(self):
        self.nlp = spacy.blank('ur')
        self.URDU_PUNCTUATIONS = ['\u200F', '\u200f', '۔', '٫', '٪', '؟', '،', ')', '(', '{', '}', '…', '۔۔۔', '/', '?', '.', '#']
        self.stop_words = frozenset("""
            آ آئی آئیں آئے آتا آتی آتے آداب آدھ آدھا آدھی آدھے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
            اتوار ارب اربویں ارے اس اسکا اسکی اسکے اسی اسے اف افوہ الاول البتہ الثانی الحرام السلام الف المکرم ان اندر انکا انکی انکے
            انہوں انہی انہیں اوئے اور اوپر اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکبر اکثر اگر اگرچہ اگست اہاہا ایسا ایسی ایسے
            ایک بائیں بار بارے بالکل باوجود باہر بج بجے بخیر برسات بشرطیکہ بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی
            بہار بہت بہتر بیگم تاکہ تاہم تب تجھ تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی
            تھیں تھے تہائی تیرا تیری تیرے تین جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں
            جنہیں جو جہاں جی جیسا جیسوں جیسی جیسے جیٹھ حالانکہ حالاں حصہ حضرت خاطر خالی خدا خزاں خواہ خوب خود دائیں درمیان دریں
            دو دوران دوسرا دوسروں دوسری دوشنبہ دوں دکھائیں دگنا دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے
            رکھا رکھتا رکھتی رکھتے رکھنا رکھنی رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے
            ساتھ سامنے ساڑھے سب سبھی سراسر سلام سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شام شاید شکریہ صاحب صاحبہ صرف
            ضرور طرح طرف طور علاوہ عین فروری فقط فلاں فی قبل قطا لائی لائے لاتا لاتی لاتے لانا لانی لایا لو لوجی لوگوں لگ
            لگا لگتا لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمی
            محض مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مطلق مل منٹ منٹوں مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں
            نا نزدیک نما نو نومبر نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وعلیکم وغیرہ
            ولے وگرنہ وہ وہاں وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونا پونی پونے پھاگن پھر پہ پہر پہلا پہلی
            پہلے پیر پیچھے چاہئے چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چوگنی چکی چکیں چکے چہارشنبہ چیت ڈالنی
            ڈالنے ڈالے کئے کا کاتک کاش کب کبھی کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کل کم
            کن کنہیں کو کوئی کون کونسا کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے
            گئی گئے گا گرما گرمی گنا گو گویا گھنٹا گھنٹوں گھنٹے گی گیا ہائیں ہائے ہاڑ ہاں ہر ہرچند ہرگز ہزار ہفتہ ہم
            ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی
            ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
        """.split())
        self.abusive_words = set([
            "گدھا","ارے بہنچود"," بہن چودوں","گ * * * پھٹ جائے ","ب * * * * * *","بہنچود", "یار بہن لوڈو","ننگی کے بچے ","چھوڑو مادر چودو","فائر گانڈو", "گدھے","ابے ماں کے ل * * *", "حرام زادہ","", "حرام زادی", "حرامی", "کمینہ", "کمینے", "کمینی", "کتی", "کتا", 
            "کتے", "پھدی", "لنڈ","بہن کی چ * *", "ماں چود", "اُلّو", "بسرڈ", "بھوسڑی", "کنجر", "چوتیا", "ماں چود", "ماں چود", "ماچود","الو" 
            "ماں چود", "میادا", "میادا", "اُلّو", "اُلّو", "اُلّو", "اُلّو کی پتی", "اُلّو", "اُلّو کا پٹھا","ر * * * کے بچے","ر * * * ",
            "اُلّو کا پٹھا", "اُلّو کا پتا", "بسرڈ چودی", "بھوسڑی", "بوسڑے کے", "بوسڑی کے", "بوسڑے کے", "بوسڑی کے","بہن چودوں" 
            "بوسڑی کے","گانڈو","کتیے","چوتیے","بھڑوے","ر * * * کے بچے","ماں کی چ * * ہندی کے بچے","پین دا پھدا پھدا پھدا","بہن دا پھدا","تیری پین دا پھدا پین چودا","تیری این دا پھدا"," دا پھدا","چنال کی اولاد کی چ * * اس کی ماں کے ل * * * لگا","چنال کی اولاد کیا","ماں چودوں بھڑکی کی","چدا","چوتیا","اپنی ماں چدا رہے ہو","گ * * * میں ڈالو پولیس","گ * * * مرا رہے ہیں بہن کے لؤڑے","ب * * * * * * سمجھ سے باہر ہے "," ب * * * * * * تو", "بوسڑے کے","ر * * * کے بچے","ابے بہن کے لؤڑے"," بہن کے ل * * * ماں چدا رہا ہوں","مادرچود"," گ * * * مرا رہے ہو", "حرام", "حرامی","بھڑوے مادرچود"," بہنچود","بہن کی لوڑی ","ارے بہن کے لؤڑے","گ * * * مرانے","ماں کو چودوں","بہنچود ر * * * کا بچہ"
        ])

    def remove_punctuations(self, text):
        return ''.join([char if char not in self.URDU_PUNCTUATIONS else ' ' for char in text])

    def remove_stopwords(self, text):
        return " ".join(word for word in text.split() if word not in self.stop_words)

    def lemmatize(self, text):
        lemme_str = ""
        temp = lemmatizer.lemma_lookup(text)
        for t in temp:
            lemme_str += t[0] + " "
        return lemme_str.strip()

    def normalize_text(self, text):
        text = remove_accents(text)
        text = normalize_whitespace(text)
        text = replace_currency_symbols(text)
        text = replace_emails(text)
        text = replace_numbers(text)
        text = replace_urls(text)
        return text

    def preprocess(self, text):
        text = self.remove_punctuations(text)
        text = " ".join([token.text for token in self.nlp(text)])
        text = self.normalize_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text

    def detect_abusive_words(self, text):
        for word in text.split():
            if word in self.abusive_words:
                return True
        return False

    def convert_to_wav(self, input_file, output_file):
        if input_file.endswith(".wav"):
            return input_file
        else:
            audio = AudioSegment.from_file(input_file)
            audio.export(output_file, format="wav")
            return output_file

    def recognize_and_preprocess_audio(self, audio_file):
        wav_file = self.convert_to_wav(audio_file, "output.wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ur-PK")
            preprocessed_text = self.preprocess(text)
            print("Preprocessed Transcription:", preprocessed_text)
            return preprocessed_text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None


def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio, language="ur-PK")
        preprocessed_text = preprocessor.preprocess(text)
        print("Preprocessed Transcription:", preprocessed_text)

        if preprocessor.detect_abusive_words(preprocessed_text):
            print("Abusive Sentence🚫")
        else:
            X_sentence = cv_loaded.transform([preprocessed_text])
            prediction = model.predict(X_sentence)

            if prediction[0] == 1:
                print("Negative Sentence😠")
            elif prediction[0] == 0:
                print("Positive Sentence😊")
            else:
                print("Neutral Sentence😐")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


cv_loaded = load('countvectorizer_updateddr4.joblib')
model = load('sentiNB_updateddr4.joblib')

preprocessor = UrduTextPreprocessor()

def main():
    option = st.input("Select mode: (1) File-based Audio Processing (2) Real-time Audio Processing: ")

    if option == '1':
        audio_file_path = st.input("Enter the path to the audio file: ")
        preprocessed_text = preprocessor.recognize_and_preprocess_audio(audio_file_path)
        print("Preprocessed Text:", preprocessed_text)

        if preprocessed_text:
            if preprocessor.detect_abusive_words(preprocessed_text):
                print("Abusive Sentence🚫")
            else:
                X_sentence = cv_loaded.transform([preprocessed_text])
                prediction = model.predict(X_sentence)

                if prediction[0] == 1:
                    print("Negative Sentence😠")
                elif prediction[0] == 0:
                    print("Positive Sentence😊")
                else:
                    print("Neutral Sentence😐")

    elif option == '2':
        recognizer = st.sr.Recognizer()
        mic = st.sr.Microphone()

        stop_listening = recognizer.listen_in_background(mic, callback)

        print("Listening for incoming audio... Press Ctrl+C to stop.")

        import time
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_listening(wait_for_stop=False)
            print("Stopped listening.")

if __name__ == "__main__":
    main()
