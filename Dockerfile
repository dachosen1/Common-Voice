FROM python:3.8.3

WORKDIR /usr/src/app

COPY commonvoice/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./run.py" ]