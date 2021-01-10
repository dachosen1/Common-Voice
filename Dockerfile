FROM python:3.7.6-slim-buster

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED True

RUN apt-get update -qq \
 && apt-get install -qqy --no-install-recommends \
      ffmpeg \
      libportaudio2 \
      libportaudiocpp0 \
      libsndfile1-dev \
      portaudio19-dev \
      pulseaudio \
      python3-pyaudio \
      gcc \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1000 ml \
 && adduser --gecos "" \
      --home /usr/src/app \
      --shell /bin/bash \
      --uid 1000 \
      --gid 1000 \
      --disabled-password \
      ml \
 && adduser ml adm \
 && adduser ml audio \
 && adduser ml pulse \
 && adduser ml voice

ENV HOME /usr/src/app
ENV PATH="/usr/src/app/.local/bin:$PATH"

WORKDIR $HOME

COPY --chown=ml:ml . .

#COPY ./requirements.txt .

RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install pyaudio
RUN pip3 install -r requirements.txt

RUN mkdir -p .local/bin .config .cache

USER ml

ENV PATH="/usr/src/app/.local/bin:$PATH"

RUN chmod +x /usr/src/app/run.sh

ENTRYPOINT /usr/src/app/run.sh