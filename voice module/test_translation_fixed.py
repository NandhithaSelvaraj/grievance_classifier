import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
).to("cuda")

print("Model ready on GPU!")

# This is the piece that was missing — it handles script normalization,
# number/date formatting, and correct tag placement before tokenization.
ip = IndicProcessor(inference=True)

src_lang = "tam_Taml"
tgt_lang = "eng_Latn"

# Plain sentence — no manually typed language tags here, the processor adds them.
input_sentences = ["எங்க ஸ்ட்ரீட்டில் மூணு நாளா தண்ணி வரல"]

# 1. Preprocess (normalize script, insert tags correctly)
batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# 2. Tokenize the *preprocessed* batch
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
).to("cuda")

# 3. Generate
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
    )

# 4. Decode tokens back to text
decoded = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

# 5. Postprocess (undoes the normalization from step 1, e.g. restores numerals)
translations = ip.postprocess_batch(decoded, lang=tgt_lang)

print("Tamil:", input_sentences[0])
print("English:", translations[0])