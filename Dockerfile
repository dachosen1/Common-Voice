FROM python:3.7.9

WORKDIR /usr/src/app

RUN	apt-get update && apt-get install -y \
	dirmngr \
	gnupg \
	--no-install-recommends \
	&& apt-get update && apt-get install -y \
	alsa-utils \
	libgl1-mesa-dri \
	libgl1-mesa-glx \
	libpulse0 \
	xdg-utils \
	libnotify-bin \
	rtkit \
	pulseaudio \
	--no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /home/ml-api-user

COPY requirements.txt ./

RUN apt-get update \
        && apt-get install libportaudio2 python3-dev libportaudiocpp0 portaudio19-dev libasound-dev libsndfile1-dev -y \
        && pip install pyaudio \
    pip install --upgrade pip \
    pip install --no-cache-dir -r requirements.txt\
    pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN for group in video audio voice pulse rtkit \
     ; do \
         adduser ml-api-user $group ; \
     done

USER ml-api-user

EXPOSE 5000

CMD ["bash", "./run.sh"]