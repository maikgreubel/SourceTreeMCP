#!/bin/sh

python -m venv .venv

. .venv/bin/activate

pip install -r requirements.txt

python server.py --base-dir $1 --transport sse
