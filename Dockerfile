FROM python:3.7.8

WORKDIR /usr/src/app


ADD commonvoice /usr/src/app/

RUN apt-get update \
        && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev -y \
        && pip3 install pyaudio

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

ADD . .

RUN chmod +x run.sh

