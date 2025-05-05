.PHONY: build-image
build-image:
	docker build -f Dockerfile.dev -t rnn .

.PHONY: servers-dev
servers-dev:
	CONFIG_DIR=$(CONFIG_DIR) CONFIG_PATH=$(CONFIG_PATH) docker compose -f docker-compose.dev.yml up

.PHONY: servers-dev-build
servers-dev-build:
	DUMP_GZIP_PATH=${(DUMP_GZIP_PATH)} CONFIG_DIR=$(CONFIG_DIR) CONFIG_PATH=$(CONFIG_PATH) docker compose -f docker-compose.dev.yml up --build
