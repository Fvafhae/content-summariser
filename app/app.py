import os
import shutil

import gradio as gr

import audio
import video
import text_sum


def text_to_sum(text, sl):

    sum = text_sum.pegasus_sum(text, sl)
    return sum

def speech_to_sum(speech):

    text = audio.silence_based_conversion(speech, None)
    return text

def video_to_audio(video_file_path):

    return video.get_audio(video_file_path)

def video_to_key_frames(video_file_path):

    kf = video.generate_key_frames(video_file_path)
    return "done"

def show_images():
    
    res = []

    for filename in os.scandir("tmp/key_frames"):
        if filename.is_file():
            res.append(gr.update(visible=True, value=filename.path))

    while len(res) < 6:
        res.append(gr.update(visible=False))

    return res


if __name__ == "__main__":
    try:
        shutil.rmtree("tmp")
    except:
        pass
    os.mkdir('tmp')

    demo = gr.Blocks()

    with demo:
        # Text Summarization
        with gr.Tab("Text Summarization"):
            with gr.Row():
                with gr.Column():
                    in1 = gr.Textbox(label="Original Text", lines=10)
                    sl1 = gr.Slider(1, 5, step=1, value=3)
                    b1 = gr.Button("Create Summary")
                with gr.Column():                
                    out1 = gr.Textbox(label="Summary", lines=10)
        
        b1.click(text_to_sum, inputs=[in1,sl1], outputs=out1)

        # Audio Summarization
        with gr.Tab("Audio Summarization"):
            audio_file1 = gr.Audio()
            b2 = gr.Button("Transcribe")
            # out2 = gr.Textbox(label="Summary")
            with gr.Row():
                with gr.Column():
                    in3 = gr.Textbox(label="Original Text", lines=10, interactive=False)
                    sl2 = gr.Slider(1, 5, step=1, value=3)
                    b3 = gr.Button("Create Summary")
                with gr.Column():
                    out2 = gr.Textbox(label="Summary", lines=10)

        b2.click(speech_to_sum, inputs=audio_file1, outputs=in3)
        b3.click(text_to_sum, inputs=[in3,sl2], outputs=out2)

        # Video Summarization
        with gr.Tab("Video Summarization"):
            video_file = gr.Video()
            b4 = gr.Button("Get Audio")
            audio_file2 = gr.Audio(interactive=False)
            b5 = gr.Button("Transcribe")
            with gr.Row():
                with gr.Column():
                    in4 = gr.Textbox(label="Original Text", lines=10, interactive=False)
                    sl3 = gr.Slider(1, 5, step=1, value=3)
                    b6 = gr.Button("Create Summary")
                with gr.Column():
                    out3 = gr.Textbox(label="Summary", lines=10)
                    state = gr.Textbox(label="Summary", visible=False)
                    with gr.Row():
                        with gr.Column():
                            img1 = gr.Image(interactive=False, visible=True)
                        with gr.Column():
                            img2 = gr.Image(interactive=False, visible=True)
                        with gr.Column():
                            img3 = gr.Image(interactive=False, visible=False)
                        with gr.Column():
                            img4 = gr.Image(interactive=False, visible=False)
                        with gr.Column():
                            img5 = gr.Image(interactive=False, visible=False)
                        with gr.Column():
                            img6 = gr.Image(interactive=False, visible=False)

        b4.click(video_to_audio, inputs=video_file, outputs=audio_file2)
        b4.click(video_to_key_frames, inputs=video_file, outputs=state)
        state.change(show_images, outputs=[img1, img2, img3, img4, img5, img6])
        b5.click(speech_to_sum, inputs=audio_file2, outputs=in4)
        b6.click(text_to_sum, inputs=[in4,sl3], outputs=out3)

    demo.launch()
