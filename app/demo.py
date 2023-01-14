import gradio as gr

import audio
import text_sum

TTS_MODEL = audio.load_model()
TS = text_sum.load_model()

def text_to_sum(text):
    global TS 

    sum = text_sum.ktrain_sum(text, TS)
    return sum

def speech_to_sum(speech):
    global TTS_MODEL

    text = audio.silence_based_conversion(speech, TTS_MODEL)
    sum = text_to_sum(text) 
    return sum

# def text_to_sentiment(text):
#     return classifier(text)[0]["label"]


demo = gr.Blocks()

with demo:
    gr.Markdown(
    """
    # Text Summarization
    """)
    in1 = gr.Textbox(label="dsad")
    b1 = gr.Button("Create Summary")
    out1 = gr.Textbox()

    gr.Markdown(
    """
    # Audio Summarization
    """)
    audio_file = gr.Audio()
    b2 = gr.Button("Create Audio Summary")
    out2 = gr.Textbox()

    b1.click(text_to_sum, inputs=in1, outputs=out1)
    b2.click(speech_to_sum, inputs=audio_file, outputs=out2)

demo.launch()
