FROM python:3.7.9-slim-buster AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED True

WORKDIR /usr/src/app

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

COPY . .

RUN pip3 install --no-cache-dir torch==1.6.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT /usr/src/app/run.sh

FROM debian:stable-slim

RUN addgroup --gid 1001 pulse
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


COPY --from=builder /usr/src/app .

RUN mkdir -p .local/bin .config .cache

RUN mkdir -p /run/user/1000 \
 && chown ml:ml /run/user/1000

USER ml

ENV PATH="/usr/src/app/.local/bin:$PATH"

COPY --chown=ml:ml . .

EXPOSE 8080