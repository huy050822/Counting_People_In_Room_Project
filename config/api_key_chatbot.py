from dotenv import load_dotenv
from dataclasses import dataclass
import os

@dataclass
class Gemini():
    api_key : str

def gemini_set_up():
    load_dotenv()

    config = {
        "gemini": Gemini(
            api_key = os.getenv("GEMINI_API_KEY")
        )
    }

    return config

if __name__ == "__main__":
    gemini_set_up()