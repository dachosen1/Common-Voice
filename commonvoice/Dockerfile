FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf
COPY ./default.conf /etc/nginx/conf.d/default.conf

COPY server /etc/nginx/

COPY ./api/templates /var/www/commvoice.me

#COPY cert /etc/nginx/commvoice_me_cert

# Generate Diffie-Hellman keys
RUN openssl dhparam -out /etc/nginx/dhparam.pem 2048


