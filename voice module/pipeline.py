import sys
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

# ---------------------------------------------------------------------------
# Language code mapping: Whisper's 2-letter codes -> IndicTrans2's tags.
# Add more rows here if your team wants to support additional languages
# (check IndicTrans2's model card on Hugging Face for the full tag list).
# ---------------------------------------------------------------------------
WHISPER_TO_INDICTRANS = {
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "hi": "hin_Deva",   # Hindi
    "kn": "kan_Knda",   # Kannada
    "ml": "mal_Mlym",   # Malayalam
    "mr": "mar_Deva",   # Marathi
    "bn": "ben_Beng",   # Bengali
}

TRANSLATION_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"


class VoiceLanguagePipeline:
    def __init__(self):
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("large")

        print("Loading IndicTrans2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL_NAME, trust_remote_code=True
        )

        print("Loading IndicTrans2 model onto GPU...")
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_NAME, trust_remote_code=True
        ).to("cuda")

        self.ip = IndicProcessor(inference=True)
        print("Pipeline ready.\n")

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """Audio file -> (native_text, whisper_language_code)."""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip(), result["language"]

    def translate(self, native_text: str, whisper_lang_code: str) -> str:
        """Native-language text -> English text, via IndicTrans2."""
        src_lang = WHISPER_TO_INDICTRANS.get(whisper_lang_code)
        if src_lang is None:
            raise ValueError(
                f"No IndicTrans2 mapping for Whisper language code "
                f"'{whisper_lang_code}'. Add it to WHISPER_TO_INDICTRANS."
            )
        tgt_lang = "eng_Latn"

        batch = self.ip.preprocess_batch(
            [native_text], src_lang=src_lang, tgt_lang=tgt_lang
        )
        inputs = self.tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_tokens = self.translation_model.generate(
                **inputs, max_length=256, num_beams=5, num_return_sequences=1
            )

        decoded = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translations = self.ip.postprocess_batch(decoded, lang=tgt_lang)
        return translations[0]

    def process(self, audio_path: str) -> dict:
        """Full pipeline: audio file -> dict matching the shared complaints schema."""
        native_text, whisper_lang = self.transcribe(audio_path)
        english_text = self.translate(native_text, whisper_lang)

        return {
            "input_type": "voice",
            "original_text": native_text,
            "detected_language": whisper_lang,
            "translated_text": english_text,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py path\\to\\audio_file")
        sys.exit(1)

    audio_file = sys.argv[1]

    pipeline = VoiceLanguagePipeline()
    result = pipeline.process(audio_file)

    print("--- RESULT ---")
    print(f"Detected language: {result['detected_language']}")
    print(f"Original text:     {result['original_text']}")
    print(f"Translated text:   {result['translated_text']}")