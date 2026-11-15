"""Start the local bottle-inspection website and open it in Chrome."""
import argparse
import os
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def open_when_ready(url):
    for _ in range(80):
        try:
            with urllib.request.urlopen(url + "/api/status", timeout=1):
                break
        except OSError:
            time.sleep(0.25)
    else:
        return
    for base in (os.environ.get("PROGRAMFILES", ""), os.environ.get("PROGRAMFILES(X86)", ""), os.environ.get("LOCALAPPDATA", "")):
        executable = Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if base and executable.is_file():
            webbrowser.BackgroundBrowser(str(executable)).open(url)
            return
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Web kiểm tra chai nhựa bằng mô hình đã huấn luyện")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    import uvicorn
    from web_api import app

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(f"http://127.0.0.1:{args.port}",), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
