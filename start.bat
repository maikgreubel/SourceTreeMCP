rem This file starts the MCP server for the Source Tree Server.

pip install -r requirements.txt

python server.py --base-dir %1 --transport sse