import gradio as gr
from config.gradio_config import Interface_config
#Gemini
from services.gemini_services import Gemini_chatbot
#


gemini_chat = Gemini_chatbot()
def chat_users(message, history):
    if history is None:
        history = []

    bot_reply = gemini_chat.chat(message, history)

    history.append({
        "role": "user",
        "content": message
    })
    history.append({
        "role": "assistant",
        "content": bot_reply
    })

    return history, ""



def clear():
    return [], "" 

def run_gradio_app():
    #Create a Blocks type 
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
        #Create Chatbot type
        chatbot = gr.Chatbot(
            avatar_images=(
                "ui/user.png",
                "ui/robot.png",
            ),
            height=600,
        )
        #Input box
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Please input...",
                scale=20,
            )
        #Button (send vs clear)
        with gr.Row():
            gr.Column(scale=9)
            send_button = gr.Button("Send", min_width=80)
            clear_button = gr.Button("Clear", min_width=80)
        #Submit (or send)
        msg.submit(
            chat_users,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        #Send button function
        send_button.click(
            chat_users,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        #Clear button function
        clear_button.click(
            clear,
            outputs=[chatbot, msg]
        )
    bl.launch(**Interface_config("basic"))
