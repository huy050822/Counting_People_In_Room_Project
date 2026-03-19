from google import genai
from config.logging_config import logger_config
from queue import Queue
import threading

logger = logger_config("services.gemini", "services.log")
request_queue = Queue()

class Gemini_chatbot:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.client = genai.Client() #Client initialize
        self.model_name = model_name
        #Use function hasattr for check attribute from one object 
        if not hasattr(self.__class__, "_worker_started"): #=> Object = Gemini_chatbot ; attribute = worker_started 
            self._start_worker() 
            self.__class__.woker_started = True 

    def chat(self, user_message, gradio_history):
        logger.info("Gemini chat request received")
        prompt = self._build_prompt(user_message, gradio_history)
        result_queue = Queue(maxsize=5)
        request_queue.put((prompt, result_queue))
        result = result_queue.get()
        return result

    def _build_prompt(self, user_message, history):
        prompt = ""
        for msg in history[-3:]:  
            role = msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n"
        prompt += f"User: {user_message}\nAssistant:"
        return prompt
    
    def _start_worker(self):
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()
        logger.info("Gemini worker started")

    def _worker(self):
        while True:
            prompt, result_queue = request_queue.get()
            try:
                logger.info("Worker calling Gemini API")
                res = self._call_gemini(prompt)
                result_queue.put(res.text)
                logger.info("Worker finished Gemini request")
            except Exception:
                logger.error(f"Gemini worker error", exc_info=True)
                result_queue.put("Busy System, Please Try Again Later")
            finally:
                request_queue.task_done()

    def _call_gemini(self, prompt, retry=3):
        for i in range(retry):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            except Exception:
                logger.error(f"Gemini API error {i+1} times", exc_info=True)
        raise RuntimeError("Gemini API failed after retries")
