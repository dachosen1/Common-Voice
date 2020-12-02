FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf
COPY ./default.conf /etc/nginx/conf.d/default.conf

COPY server /etc/nginx/

# Generate Diffie-Hellman keys
RUN openssl dhparam -out /etc/nginx/dhparam.pem 2048


