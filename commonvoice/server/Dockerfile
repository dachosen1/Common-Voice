FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf

COPY . /etc/nginx/

# Generate Diffie-Hellman keys
RUN openssl dhparam -out /etc/nginx/dhparam.pem 2048

# Create a common ACME-challenge
RUN mkdir -p /var/www/_letsencrypt
RUN chown www-data /var/www/_letsencrypt

# Obtain SSL certificates from Let's Encrypt using Certbot:
RUN apt-get update
RUN apt-get -y install python-certbot-nginx
RUN sed -i -r 's/(listen .*443)/\1;#/g; s/(ssl_(certificate|certificate_key|trusted_certificate) )/#;#\1/g' /etc/nginx/sites-available/commvoice.me.conf
RUN certbot certonly --webroot -d commvoice.me -d www.commvoice.me --email anderson.nelson1@gmail.com -w /var/www/_letsencrypt -n --agree-tos --force-renewal

# Uncomment SSL related directives in the configuration:
RUN sed -i -r 's/#?;#//g' /etc/nginx/sites-available/commvoice.me.conf

# Configure Certbot to reload NGINX when it successfully renews certificates:
RUN echo -e '#!/bin/bash\nnginx -t && systemctl reload nginx' | tee /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh
RUN chmod a+x /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh
