FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY 40-password.sh /docker-entrypoint.d/40-password.sh
RUN chmod 755 /docker-entrypoint.d/40-password.sh
