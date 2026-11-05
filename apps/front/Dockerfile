FROM node:22-alpine AS build
WORKDIR /app
COPY apps/front/package*.json ./
RUN npm ci
COPY apps/front/ ./
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine
COPY apps/front/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8501
