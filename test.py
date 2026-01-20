from services.gemini_services import Gemini_chatbot
bot = Gemini_chatbot()

history = [
    ("hello", "Hi! How can I help you?"),
    ("what is AI?", "AI is artificial intelligence.")
]

print(bot.chat("explain again shortly", history))
