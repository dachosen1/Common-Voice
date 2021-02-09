## Common Voice
[![CodeFactor](https://www.codefactor.io/repository/github/dachosen1/common-voice/badge)](https://www.codefactor.io/repository/github/dachosen1/common-voice)
[![CircleCI](https://circleci.com/gh/dachosen1/Common-Voice.svg?style=svg)](https://circleci.com/gh/dachosen1/Common-Voice)

**Live Deployment**: [Commvoice](https://commvoice.me/)

### Data  
The data for this project is sourced from [Common Voice](https://commonvoice.mozilla.org/en), which is a crowdsourcing project started by Mozilla to create a free database for speech recognition software. The project is supported by volunteers who record sample sentences with a microphone and review recordings of other users. The transcribed sentences will be collected in a voice database available under the public domain license CC0. This license ensures that developers can use the database for voice-to-text applications without restrictions or costs. Common Voice appeared as a response to the language assistants of large companies such as Amazon Echo, Siri or Google Assistant

# Overview
The goal for this project is to create an end to end machine learning appliacation that records and processes audio in real time and stream prediction via a socket API. There's a 1 second delay delay between the audio recording and the output prediction.

![Common Voice](https://user-images.githubusercontent.com/40616129/107158901-bd05d980-6941-11eb-92b5-12d7c001d5a1.PNG)

The application generates prediction in 3 categories: Gender, Age and Country of Origin. 

## Todo:
Traing and implement models for Country and Age

# Getting Started: 
## Train Model 

Modify the Data Directory to your own direcory 

```
class DataDirectory:
    DATA_DIR = r"C:\Users\ander\Documents\common-voice-data"
    DEV_DIR = r"C:\Users\ander\Documents\common-voice-dev"
    CLIPS_DIR = r"C:\Users\ander\Documents\common-voice-data\clips"
```

Run machine learning pipeline
```
python run_pipeline.py
```
## Deploy Model
Windows Machine 

```
pip3 install -r requirements.txt
python3 run_app.py
```

Linux 
Install the below on the server prior to running the docker images 

**Step 1:** Install Docker and Docker  Compose 

```
# Install Docker compose 
sudo curl -L "https://github.com/docker/compose/releases/download/1.28.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Install Docker 
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

**Step 2:** Install Nginx 

```
# Install NGINX 
sudo apt install certbot python3-certbot-nginx
sudo nano /etc/nginx/sites-available/commvoice.me
...
server_name WEBSITE_NAME WEBSITE_NAME;
...

```

**Step 3:** Install Certbot 

```
# Install Certbot 
sudo certbot --nginx -d commvoice.me -d www.commvoice.me

# Install Dhparam 
openssl dhparam -out /etc/nginx/dhparam.pem 2048

# Install Certbot Auto Renew 
systemctl status certbot.timer
```

**Step 4:** Install and audio Drive 
Enabled a snd-aloop modules 

```
modprobe snd-aloop
```
The below devices should have been added to you dev/snd directory.

```
ls /dev/snd/

    -  pcmC0D0c
    -  pcmC0D0p
    -  pcmC0D1c
```

**Step 5:** Run Docker Compose 

```
docker-commpose up --build
```

