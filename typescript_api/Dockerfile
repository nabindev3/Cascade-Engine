FROM node:22-slim

WORKDIR /app

COPY package.json .
RUN npm install

COPY tsconfig.json .
COPY src/ ./src/

RUN npx tsc

EXPOSE 3000

CMD ["node", "dist/server.js"]
