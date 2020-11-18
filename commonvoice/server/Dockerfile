FROM nginx:1.19.3

RUN rm /etc/nginx/conf.d/default.conf
COPY default.conf /etc/nginx/conf.d/

COPY nginx.conf /etc/nginx/
COPY sites-available/commvoice.me.conf /etc/nginx/sites-available/
COPY nginxconfig.io /etc/nginx/nginxconfig.io/

RUN openssl dhparam -out /etc/nginx/dhparam.pem 2048

RUN mkdir -p /var/www/_letsencrypt

RUN chown www-data /var/www/_letsencrypt
