import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

# Load AI models
chatbot = pipeline("text-generation", model="facebook/opt-1.3b")
video_generator = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

# AI Chatbot Function
def ai_chat(prompt):
    response = chatbot(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# AI Video Generation
def generate_video(prompt):
    image = video_generator(prompt).images[0]
    image.save("generated_image.png")

    # Convert image to a short video
    clip = ImageSequenceClip(["generated_image.png"] * 30, fps=10)
    clip.write_videofile("generated_video.mp4", codec="libx264")

    return "generated_video.mp4"

# Image Animation Function
def animate_image(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    video = cv2.VideoWriter("animated_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for i in range(20):  
        frame = cv2.addWeighted(img, 1 - (i / 20), np.zeros_like(img), (i / 20), 0)
        video.write(frame)

    video.release()
    return "animated_video.mp4"

# UI Layout
with gr.Blocks() as FlyingTalk:
    gr.Markdown("<h1 style='color: white; position: absolute; top: 10px; left: 10px;'>FlyingTalk</h1>")
    
    with gr.Row():
        chat_history = gr.Chatbot(label="FlyingTalk Chat", show_label=False)
    
    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your message...", show_label=False)
        send_button = gr.Button("Send")

    send_button.click(ai_chat, inputs=user_input, outputs=chat_history)
    
    with gr.Row():
        video_prompt = gr.Textbox(placeholder="Enter a text prompt for video", label="Text-to-Video")
        generate_button = gr.Button("Generate Video")
        video_output = gr.Video()

    generate_button.click(generate_video, inputs=video_prompt, outputs=video_output)
    
    with gr.Row():
        image_input = gr.File(label="Upload an Image for Animation")
        animate_button = gr.Button("Animate")
        animation_output = gr.Video()

    animate_button.click(animate_image, inputs=image_input, outputs=animation_output)

# Launch Gradio Interface
FlyingTalk.launch(server_name="0.0.0.0", server_port=8080)
