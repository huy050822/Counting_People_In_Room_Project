import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

LOG_DIR = "logs"  #Set folder name
os.makedirs(LOG_DIR, exist_ok=True) #Create
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
#Config để sau scale code ra dễ
CONFIG = {
    "rotating" : {
        "class": RotatingFileHandler,
        "kwargs": {
            "maxBytes": 5*1024*1024,
            "backupCount": 5,
            "encoding": "utf-8"
        }
    },
    "timed": {
        "class": TimedRotatingFileHandler,
        "kwargs": {
            "when": "midnight",
            "interval": 1,
            "backupCount": 5,
            "encoding": "utf-8"
        }
    }
}
#Create def handlers 
def handlers_config(file, handler_type= "rotating"):
    default = CONFIG[handler_type] #Set default handler_type from config dict
    formatter = logging.Formatter(FORMAT)
    #Little apply => just use this form for another project 
    handler = default["class"](
        os.path.join(LOG_DIR,file),
        **default["kwargs"]
    )
    #Apply formatter
    handler.setFormatter(formatter)
    return handler
#Create def write logger 
def logger_config(name, file_name, level=logging.INFO, handler_type= "rotating"):
    logger = logging.getLogger(name) # == file's name <app.py> => <app.logs>
    logger.setLevel(level) 
    if not logger.handlers:
        return logger #Avoid write over the old file
    
    handler = handlers_config(file_name, handler_type) #Call def handlers_config and use it such as default
    logger.addHandler(handler) #Run this handler => if not just stay as long as they used to
    return logger