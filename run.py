"""
run.py

One-command launcher for the full Waste Segregation demo: it starts the
FastAPI backend (which also serves the Three.js frontend at `/`) and opens the
3D dashboard in your browser once the server is confirmed healthy.

Because the frontend (`web/index.html`) talks to the API with same-origin
relative paths, the backend and frontend are served by a single uvicorn
process -- there is no second server to start. This script therefore launches
that server and pops open the dashboard the moment it is ready.

Usage:
    uv run --extra serve python run.py
    uv run --extra serve python run.py --port 9000 --no-browser
    uv run --extra serve python run.py --reload        # auto-reload on code edits

Then:
    Dashboard : http://127.0.0.1:8000/
    API docs  : http://127.0.0.1:8000/docs
Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


def _serve_extra_available() -> bool:
    """True if the optional `serve` dependencies are installed."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def _wait_until_ready_then_open(url: str, health_url: str,
                                timeout: float = 20.0) -> None:
    """Poll the server in a background thread; open the browser once it answers.

    Runs off the main thread so uvicorn can own the main thread (and thus
    handle Ctrl+C cleanly).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status < 500:
                    break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.25)
    else:
        print(f"[run] Server did not become ready within {timeout:.0f}s; "
              f"open {url} manually.", flush=True)
        return

    print(f"\n[run] Backend + frontend are live.\n"
          f"[run]   Dashboard : {url}\n"
          f"[run]   API docs  : {url}docs\n"
          f"[run] Opening the dashboard in your browser...\n", flush=True)
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - headless / no browser available
        print(f"[run] Could not auto-open a browser; visit {url} yourself.",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Waste Segregation backend + 3D frontend together."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open the dashboard in a browser.")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload the server on source changes (dev).")
    args = parser.parse_args()

    if not _serve_extra_available():
        sys.exit(
            "The serving dependencies (FastAPI + uvicorn) are not installed.\n"
            "Run this launcher with the `serve` extra, e.g.:\n\n"
            "    uv run --extra serve python run.py\n"
        )

    import uvicorn

    if not _port_is_free(args.host, args.port):
        sys.exit(
            f"Port {args.port} on {args.host} is already in use. "
            f"Stop the other process or pass --port <n>."
        )

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        health_url = f"{url}docs"  # a cheap, always-present endpoint
        threading.Thread(
            target=_wait_until_ready_then_open,
            args=(url, health_url),
            daemon=True,
        ).start()

    print(f"[run] Starting server on {url} (Ctrl+C to stop)...", flush=True)
    try:
        uvicorn.run("serve:app", host=args.host, port=args.port,
                    reload=args.reload, log_level="info")
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[run] Shutting down. Bye.", flush=True)


if __name__ == "__main__":
    main()
