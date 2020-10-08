FROM python:3.7.8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
 && apt-get install -qqy --no-install-recommends \
      ffmpeg \
      libportaudio2 \
      libportaudiocpp0 \
      libsndfile1-dev \
      portaudio19-dev \
      pulseaudio \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pyaudio
RUN pip3 install --no-cache-dir torch==1.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

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

RUN mkdir -p /run/user/1000 \
 && chown ml:ml /run/user/1000

WORKDIR /usr/src/app

USER ml

RUN mkdir -p .local/bin .config .cache

ENV PATH="/usr/src/app/.local/bin:$PATH"

COPY --chown=ml:ml . .

RUN pip3 install --no-cache-dir -r requirements.txt
RUN chown /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/

ENTRYPOINT /usr/src/app/run.sh