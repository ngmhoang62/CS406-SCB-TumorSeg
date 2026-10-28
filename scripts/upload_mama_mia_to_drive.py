"""
Robust, Resumable Google Drive Uploader for MAMA-MIA Raw Dataset (CS406-IPA).

Key Features:
- Prevents Windows system sleep/standby during upload via SetThreadExecutionState.
- Persists session URL in scripts/.upload_session.json for crash recovery.
- Auto-refreshes OAuth2 access token from antigravity token.json.
- Queries server for exact byte offset on every chunk / retry to prevent range mismatch.
- Handles connection timeouts & network drops with exponential backoff auto-retry.
- Automatically deletes local mama_mia_raw.zip upon 100% completion to free 87.8 GB on drive D:.
"""

import os
import sys
import time
import json
import ctypes
from pathlib import Path
import requests
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "mama_mia_raw.zip"
SESSION_FILE = PROJECT_ROOT / "scripts" / ".upload_session.json"
TOKEN_PATH = Path.home() / ".gemini" / "antigravity" / "plugin_data" / "gdrive" / "token.json"
FOLDER_ID = "1_R2FBnFZf3N0kXWirW1S9bpbid-Jt14M"
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB (multiple of 256 KB)

# Windows API flags to prevent sleep
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        print("[Power] Windows Sleep prevention enabled during upload.")
    except Exception as e:
        print(f"[Power] Warning: could not set sleep prevention: {e}")


def allow_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[Power] Windows Sleep settings restored.")
    except Exception:
        pass


def get_fresh_access_token() -> str:
    with open(TOKEN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    token_info = data.get("token", {})
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    refresh_token = token_info.get("refresh_token")

    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_or_create_session(file_size: int) -> str:
    # 1. Check existing session file
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            url = saved.get("session_url")
            # Probe session
            probe = requests.put(url, headers={"Content-Range": f"bytes */{file_size}"}, timeout=20)
            if probe.status_code == 308:
                print(f"[Session] Reused active session: {url[:80]}...")
                return url
        except Exception as e:
            print(f"[Session] Saved session expired or invalid ({e}), creating a fresh one.")

    # 2. Create fresh session
    token = get_fresh_access_token()
    metadata = {
        "name": "mama_mia_raw.zip",
        "parents": [FOLDER_ID],
    }
    resp = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/zip",
            "X-Upload-Content-Length": str(file_size),
        },
        json=metadata,
        timeout=30,
    )
    resp.raise_for_status()
    session_url = resp.headers["Location"]
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump({"session_url": session_url, "created_at": time.time()}, f)
    print(f"[Session] Created new session: {session_url[:80]}...")
    return session_url


def query_server_offset(session_url: str, file_size: int) -> int:
    resp = requests.put(session_url, headers={"Content-Range": f"bytes */{file_size}"}, timeout=30)
    if resp.status_code == 308:
        rng = resp.headers.get("Range")
        if rng and rng.startswith("bytes=0-"):
            return int(rng.split("-")[1]) + 1
        return 0
    elif resp.status_code in (200, 201):
        return file_size
    else:
        raise RuntimeError(f"Unexpected status querying offset: {resp.status_code} - {resp.text}")


def upload_dataset():
    print("=" * 80)
    print("MAMA-MIA RAW DATASET BULLETPROOF UPLOADER (CS406-IPA)")
    print("=" * 80)

    if not ZIP_PATH.exists():
        print(f"Error: {ZIP_PATH} not found.")
        sys.exit(1)

    file_size = ZIP_PATH.stat().st_size
    print(f"File: {ZIP_PATH} ({file_size / (1024**3):.2f} GB)")

    prevent_sleep()
    try:
        session_url = get_or_create_session(file_size)
        current_offset = query_server_offset(session_url, file_size)
        print(f"Server starting offset: {current_offset / (1024**3):.2f} GB / {file_size / (1024**3):.2f} GB ({current_offset / file_size * 100:.2f}%)")

        with open(ZIP_PATH, "rb") as f:
            with tqdm(total=file_size, initial=current_offset, unit="B", unit_scale=True, unit_divisor=1024, desc="Uploading") as pbar:
                while current_offset < file_size:
                    chunk_len = min(CHUNK_SIZE, file_size - current_offset)
                    end_byte = current_offset + chunk_len - 1

                    f.seek(current_offset)
                    chunk_data = f.read(chunk_len)

                    # Send with retry loop
                    sent = False
                    backoff = 3
                    while not sent:
                        try:
                            resp = requests.put(
                                session_url,
                                headers={
                                    "Content-Length": str(len(chunk_data)),
                                    "Content-Range": f"bytes {current_offset}-{end_byte}/{file_size}",
                                },
                                data=chunk_data,
                                timeout=120,
                            )
                            if resp.status_code in (200, 201):
                                pbar.update(len(chunk_data))
                                print("\n\n" + "=" * 80)
                                print("UPLOAD COMPLETED SUCCESSFULLY 100%!")
                                print("=" * 80)
                                print("Response:", resp.json())
                                on_success()
                                return
                            elif resp.status_code == 308:
                                rng = resp.headers.get("Range")
                                if rng and rng.startswith("bytes=0-"):
                                    server_end = int(rng.split("-")[1])
                                    advanced = server_end + 1 - current_offset
                                    pbar.update(advanced)
                                    current_offset = server_end + 1
                                else:
                                    pbar.update(len(chunk_data))
                                    current_offset = end_byte + 1
                                sent = True
                            else:
                                print(f"\n[Warning] HTTP {resp.status_code}: {resp.text}. Retrying in {backoff}s...")
                                time.sleep(backoff)
                                backoff = min(60, backoff * 2)
                                current_offset = query_server_offset(session_url, file_size)
                                # Re-slice chunk matching new offset
                                chunk_len = min(CHUNK_SIZE, file_size - current_offset)
                                end_byte = current_offset + chunk_len - 1
                                f.seek(current_offset)
                                chunk_data = f.read(chunk_len)
                        except Exception as e:
                            print(f"\n[Network] {e}. Retrying in {backoff}s...")
                            time.sleep(backoff)
                            backoff = min(60, backoff * 2)
                            try:
                                current_offset = query_server_offset(session_url, file_size)
                                chunk_len = min(CHUNK_SIZE, file_size - current_offset)
                                end_byte = current_offset + chunk_len - 1
                                f.seek(current_offset)
                                chunk_data = f.read(chunk_len)
                            except Exception:
                                pass
    finally:
        allow_sleep()


def on_success():
    if SESSION_FILE.exists():
        SESSION_FILE.unlink(missing_ok=True)
    print(f"\nDeleting local archive {ZIP_PATH} to free disk space...")
    try:
        ZIP_PATH.unlink(missing_ok=True)
        print("mama_mia_raw.zip deleted successfully! Freed 87.80 GB on drive D:.")
    except Exception as e:
        print(f"Error deleting zip file: {e}")


if __name__ == "__main__":
    upload_dataset()

