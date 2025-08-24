FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web_chatbot.py", "--server.port", "8501", "--server.address", "0.0.0.0"]