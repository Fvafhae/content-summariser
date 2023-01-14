import torch

from ktrain.text.summarization import TransformerSummarizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def load_model():
    return TransformerSummarizer()

def ktrain_sum(text, ts):
    return ts.summarize(text)

def pegasus_sum(text):
    
    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    batch = tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)

    translated = model.generate(
        **batch,                 
        num_beams=4,
        length_penalty=2.0,
        max_new_tokens=142,
        min_length=42,
        no_repeat_ngram_size=3)

    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0]
