FROM python:3.8.6-slim-buster

ENV  DEBIAN_FRONTEND=noninteractive
ENV  PYTHONUNBUFFERED True

RUN  apt-get update -qq \
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

WORKDIR /usr/src/app

RUN  addgroup --gid 1000 ml \
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

COPY ./requirements.txt .

RUN  pip3 install --no-cache-dir --upgrade pip \
     torch==1.6.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \ 
     pyaudio \
     -r requirements.txt

COPY . .

RUN  mkdir -p .local/bin .config .cache \
     mkdir -p /run/user/1000 \
     && chown ml:ml /run/user/1000

USER ml

ENV  PATH="/usr/src/app/.local/bin:$PATH"

COPY --chown=ml:ml . .

RUN  chmod +x /usr/src/app/run.sh

EXPOSE 8080

ENTRYPOINT /usr/src/app/run.sh