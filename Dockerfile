FROM python:3.7

WORKDIR /usr/src/app

ENV FLASK_APP run_app.py

COPY commonvoice/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN	pip install pipwin
RUN	pipwin install pyaudio

COPY . .

RUN chmod +x run.sh

EXPOSE 5000

CMD [ "python", "./run_app.py"]
