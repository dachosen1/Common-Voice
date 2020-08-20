FROM python:3.8

WORKDIR /usr/src/app

ENV FLASK_APP run_app.py

COPY commonvoice/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.0.1 torchvision==0.2.2
RUN	pip install pipwin
RUN	pipwin install pyaudio

RUN chmod +x run.sh

COPY . .

EXPOSE 5000

CMD [ "python", "./run_pipeline.py" ]
