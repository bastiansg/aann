core-build:
	docker compose build aann-core

core-run:
	docker compose run aann-core


jupyter-build: core-build
	docker compose build aann-jupyter

jupyter-run:
	docker compose up aann-jupyter-gpu

jupyter-run-cpu:
	docker compose up aann-jupyter-cpu
