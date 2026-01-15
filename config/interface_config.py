from dotenv import load_dotenv
from dataclasses import dataclass
import os

@dataclass
class Interface():
    host : str
    port : int
    share : str

def Interface_config():
    load_dotenv()
    config = {
        "basic" : Interface(
            host = os.getenv("GRADIO_HOST"),
            port = os.getenv("GRADIO_PORT"),
            share = os.getenv("SHARE")
        )
    }
    return config