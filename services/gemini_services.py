from google import genai

class Gemini_chatbot():
    def __init__(self, api_key, model_name= "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name 

    def chat(self, user_message, history):
        messages = []
        for user, bot in history:
            messages.append({
                "role": "user", 
                "parts": [{"text": user}] 
            })
            messages.append({
                "role": "model", 
                "parts": [{"text": bot}] 
            })
        
        messages.append({
            "role": "user", 
            "parts": [{"text": user_message}] 
        })
        
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=messages
        )
        return res.text
