#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repository_root"

for obsolete in package.json package-lock.json electron.vite.config.ts electron-builder.yml tsconfig.json src/preload src/renderer browser-extension; do
  if [ -e "$obsolete" ]; then
    echo "obsolete Electron architecture remains: $obsolete" >&2
    exit 1
  fi
done

if rg -n -i --glob '!Cargo.lock' --glob '!apps/macos/Package.resolved' --glob '!check-repository.sh' \
  '(electron-vite|nodeIntegration|contextBridge|BrowserWindow)' \
  Cargo.toml crates apps proto scripts product.toml >/dev/null; then
  echo "Electron-specific code remains in the native/Rust source tree" >&2
  exit 1
fi

if rg -n --hidden --glob '!.git/**' --glob '!target/**' --glob '!apps/macos/.build/**' \
  '(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,})' . >/dev/null; then
  echo "possible credential material detected" >&2
  exit 1
fi

echo "Repository architecture and credential checks passed."
