FROM node:18-alpine

RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /app

COPY . .

RUN bun install
RUN bun run build

CMD ["bun", "run", "preview"]
