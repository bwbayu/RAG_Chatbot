FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "web_chatbot.py", "--server.port=8080", "--server.address=0.0.0.0"]