FROM nginx:alpine

COPY . /etc/nginx/

#RUN adduser --disabled-password --gecos '' www-data

RUN apk add certbot certbot-nginx
RUN apk add acf-openssl

RUN openssl dhparam -out /etc/nginx/dhparam.pem 2048
RUN mkdir -p /var/www/_letsencrypt

RUN certbot certonly --webroot -d commvoice.me --email anderson.nelson1@gmail.com -w /var/www/_letsencrypt -n --agree-tos --force-renewal

RUN sed -i -r 's/#?;#//g' /etc/nginx/sites-available/commvoice.me.conf
RUN echo -e '#!/bin/bash\nnginx -t && systemctl reload nginx' | tee /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh

RUN chmod a+x /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh