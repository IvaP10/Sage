#!/usr/bin/env python3
"""Register Sage's native messaging host for one explicitly selected extension."""
import argparse
import json
import os
from pathlib import Path
import platform
import re

parser = argparse.ArgumentParser()
parser.add_argument("extension_id", help="ID shown by Chrome/Edge for the loaded Sage companion")
parser.add_argument("--executable", type=Path, required=True)
args = parser.parse_args()
if not re.fullmatch(r"[a-p]{32}", args.extension_id):
    parser.error("Expected the 32-letter extension ID")
executable = args.executable.resolve(strict=True)
if platform.system() == "Darwin":
    data = Path.home() / "Library/Application Support/Sage"
    roots = [Path.home() / "Library/Application Support" / name / "NativeMessagingHosts" for name in ["Google/Chrome", "Microsoft Edge", "BraveSoftware/Brave-Browser"]]
elif platform.system() == "Windows":
    data = Path(os.environ["LOCALAPPDATA"]) / "Sage"
    roots = []
else:
    parser.error("This installer supports macOS and Windows")
data.mkdir(parents=True, exist_ok=True)
manifest = {"name": "com.ivanpadeliya.sage.browser", "description": "Sage local browser adapter", "path": str(executable), "type": "stdio", "allowed_origins": [f"chrome-extension://{args.extension_id}/"]}
host = data / "browser-host.json"
host.write_text(json.dumps(manifest, indent=2) + "\n")
host.chmod(0o600)
for root in roots:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "com.ivanpadeliya.sage.browser.json"
    destination.write_text(host.read_text())
    destination.chmod(0o600)
if platform.system() == "Windows":
    import winreg
    for browser in [r"Google\Chrome", r"Microsoft\Edge"]:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\{browser}\NativeMessagingHosts\com.ivanpadeliya.sage.browser") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(host))
print("Registered the selected extension. Open Sage, then click its browser companion on the tab to pair.")
