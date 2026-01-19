from dotenv import load_dotenv
import os

def Interface_config(mode= "basic"):
    load_dotenv()
    config = {
        "basic": {
            "server_name": os.getenv("GRADIO_HOST"),
            "server_port": int(os.getenv("GRADIO_PORT")),
            "share": False
        }
    }
    return config[mode]