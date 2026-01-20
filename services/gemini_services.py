from google import genai

class Gemini_chatbot:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.client = genai.Client()
        self.model_name = model_name

    def chat(self, user_message, gradio_history):
        # Gộp history thành text thuần
        prompt = ""

        for msg in gradio_history:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"

        prompt += f"User: {user_message}\nAssistant:"

        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return res.text
