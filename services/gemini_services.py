from google import genai

class Gemini_chatbot:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.client = genai.Client()
        self.model_name = model_name

    def chat(self, user_message, history):
        contents = []

        for msg in history:
            if msg["role"] == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })

        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        res = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

        return res.text

