FROM python:3.10
WORKDIR /main
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
COPY . .
CMD ["python","main.py"]