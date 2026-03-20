import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from config.logging_config import logger_config
from config.gradio_config import Interface_config
from models.data_path import Data_Path
from services.gemini_services import Gemini_chatbot
from models.model_cpv import CrowdCountingModel   

logger = logger_config("gradio_app.py", "app.log")
gemini_chat = Gemini_chatbot()
DEVICE = Data_Path.DEVICE

def load_model():
    model = CrowdCountingModel().to(DEVICE)
    checkpoint = torch.load("trained_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Crowd model loaded")
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
def chat_users(message, history):
    if history is None:
        history = []

    logger.info("Starting using Gemini to reply user")
    bot_reply = gemini_chat.chat(message, history)

    if bot_reply is None:
        bot_reply = "Gemini did not return a response."
        logger.info("Gemini did not return a response")

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_reply})

    return history, ""
def clear():
    logger.info("Clear history completely")
    return [], ""

def count_people(image):
    if image is None:
        return "No image uploaded", None

    logger.info("Start counting people")

    image = image.astype(np.uint8)
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        density = model(img_tensor)[0, 0].cpu().numpy()

    count = float(density.sum())

    density_vis = cv2.resize(density, (224, 224))
    density_vis = (density_vis - density_vis.min()) / (density_vis.max() + 1e-8)

    logger.info(f"Predicted count: {count:.2f}")

    return f"Number of People: {count:.2f}", density_vis

def run_gradio_app():
    logger.info("Interface initialized")

    with gr.Blocks(theme=gr.themes.Soft()) as bl:

        with gr.Row():
            gr.Column(scale=1)
            gr.Image(
                value="ui/logo_chatbot.png",
                show_label=False,
                height=180,
                container=False
            )
            gr.Column(scale=1)

        chatbot = gr.Chatbot(
            avatar_images=("ui/user.png", "ui/robot.png"),
            height=400,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Please input...",
                scale=20,
            )

        with gr.Row():
            send_button = gr.Button("Send")
            clear_button = gr.Button("Clear")

        gr.Markdown("## Crowd Counting")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            density_output = gr.Image(label="Density Map")

        count_output = gr.Textbox(label="Result")
        count_button = gr.Button("Count People")

        msg.submit(chat_users, inputs=[msg, chatbot], outputs=[chatbot, msg])
        send_button.click(chat_users, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_button.click(clear, outputs=[chatbot, msg])
        count_button.click(
            count_people,
            inputs=image_input,
            outputs=[count_output, density_output]
        )

    bl.launch(**Interface_config("basic"))
    logger.info("Interface opened")