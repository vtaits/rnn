FROM node:24-alpine

RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /app

RUN bun install

COPY . .

RUN bun run build

CMD ["bun", "run", "preview"]
