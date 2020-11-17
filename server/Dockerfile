FROM nginx:1.19.3

COPY nginx.conf /etc/nginx/
COPY sites-available/commvoice.me.conf /etc/nginx/sites-available/
COPY nginxconfig.io/ /etc/nginx/nginxconfig.io/

RUN openssl dhparam -out /etc/nginx/dhparam.pem 2048

RUN mkdir -p /var/www/_letsencrypt

RUN chown www-data /var/www/_letsencrypt

#RUN sed -i -r 's/(listen .*443)/\1;#/g; s/(ssl_(certificate|certificate_key|trusted_certificate) )/#;#\1/g' /etc/nginx/sites-available/commvoice.me.conf
#RUN nginx -t && systemctl reload nginx


#RUN certbot certonly --webroot -d commvoice.me -d www.commvoice.me --email info@commvoice.me -w /var/www/_letsencrypt -n --agree-tos --force-renewal

#RUN sed -i -r 's/#?;#//g' /etc/nginx/sites-available/commvoice.me.conf


#RUN echo -e '#!/bin/bash\nnginx -t' | tee /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh
#RUN sudo chmod a+x /etc/letsencrypt/renewal-hooks/post/nginx-reload.sh \
#    sudo nginx -t && sudo systemctl reload nginx

