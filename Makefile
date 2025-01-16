.PHONY: build-image
build-image:
	docker build -f Dockerfile.dev -t rnn .

.PHONY: servers-dev
servers-dev:
	CONFIG_DIR=/home/vadim/projects/rnn/configs CONFIG_PATH=letters_6x6.toml docker compose -f docker-compose.dev.yml up

