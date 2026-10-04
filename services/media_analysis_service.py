import os
import tempfile

from PIL import Image
import pytesseract

import speech_recognition as sr
from pydub import AudioSegment


# ==================================================
# TESSERACT PATH (Windows fallback)
# ==================================================
#
# If Tesseract is installed but not on the system PATH,
# pytesseract won't find it automatically. This points
# directly to the default UB-Mannheim install location.
# If you installed it somewhere else, update this path
# to match, or comment this line out entirely if
# Tesseract IS on your PATH already.

_DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_DEFAULT_TESSDATA_DIR = r"C:\Program Files\Tesseract-OCR\tessdata"

if os.path.exists(_DEFAULT_TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_PATH

if os.path.exists(_DEFAULT_TESSDATA_DIR):

    os.environ["TESSDATA_PREFIX"] = _DEFAULT_TESSDATA_DIR


# ==================================================
# IMAGE TEXT EXTRACTION (OCR)
# ==================================================

def extract_text_from_image(uploaded_file):

    """
    Extracts visible text from an uploaded image
    (e.g. a screenshot of a social media post or a
    meme with text overlay) using Tesseract OCR.
    """

    try:

        image = Image.open(uploaded_file)

        text = pytesseract.image_to_string(image)

        return text.strip()

    except Exception as error:

        raise RuntimeError(
            f"Failed to extract text from image: {error}"
        )


# ==================================================
# AUDIO / VIDEO TRANSCRIPTION
# ==================================================

def transcribe_audio_video(uploaded_file, original_filename):

    """
    Transcribes speech from an uploaded audio or video
    file into text using free speech recognition.

    Video files: the audio track is extracted first.
    Audio files: used directly.

    Requires ffmpeg to be installed and available on
    the system PATH (used internally by pydub).
    """

    suffix = os.path.splitext(original_filename)[1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as temp_input:

        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    wav_path = temp_input_path + "_converted.wav"

    try:

        # pydub + ffmpeg handles both audio and video
        # files - it extracts/decodes the audio track
        # regardless of the original container format.

        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:

            audio_data = recognizer.record(source)

        try:

            text = recognizer.recognize_google(audio_data)

        except sr.UnknownValueError:

            text = ""

        except sr.RequestError as error:

            raise RuntimeError(
                f"Speech recognition service error: {error}"
            )

        return text.strip()

    except Exception as error:

        raise RuntimeError(
            f"Failed to transcribe audio/video: {error}"
        )

    finally:

        for path in (temp_input_path, wav_path):

            if os.path.exists(path):

                try:
                    os.remove(path)
                except Exception:
                    pass