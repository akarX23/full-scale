"""
Direct entry-point for running the server as a module:

    python -m inference_server
    python -m inference_server --served-model alexnet --port 8080
"""

import uvicorn

from .config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
