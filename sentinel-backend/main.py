"""
Uvicorn entry point for the Sentinel ISS backend.

This is the module that ``uvicorn main:app`` loads.  It simply calls
``create_app()`` from the application package, which wires up all
services, middleware, and route handlers.

Usage::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload --ws-per-message-deflate false

Keep ``--ws-per-message-deflate false``: camera frames are JPEGs, which do not
compress, and uvicorn otherwise deflates every frame on the event loop.
"""

from app.main import create_app

# The ASGI application instance consumed by uvicorn.
app = create_app()
