# Text to Speech

import os
from gtts import gTTS
import pyttsx3
import playsound
import tempfile

def speak_text(text, use_online=True, lang="en"):
    if not text.strip():
        return
    try:
        if use_online:
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                playsound.playsound(fp.name)
            os.remove(fp.name)
        else:
            raise Exception("Offline Mode")
    except Exception:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("[TTS Error]", e)