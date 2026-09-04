#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repository_root"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "macOS packaging must run on macOS" >&2
  exit 1
fi

architecture=${SAGE_MACOS_ARCH:-$(uname -m)}
case "$architecture" in
  arm64|aarch64) architecture=arm64 ;;
  x86_64|amd64) architecture=x86_64 ;;
  *) echo "unsupported macOS architecture: $architecture" >&2; exit 1 ;;
esac

output_root=${SAGE_MACOS_OUTPUT_ROOT:-$repository_root/dist/macos/$architecture}
application=${SAGE_MACOS_APP_DIR:-$output_root/Sage.app}
contents="$application/Contents"
disk_image=${SAGE_MACOS_DMG_PATH:-$output_root/Sage-1.0.1-macos-$architecture-preview.dmg}

cargo build --release --workspace
swift build --package-path apps/macos -c release --product SageMac

rm -rf "$application"
mkdir -p "$contents/MacOS" "$contents/Helpers" "$contents/Resources"

cp apps/macos/.build/release/SageMac "$contents/MacOS/Sage"
cp target/release/sage-core "$contents/Helpers/sage-core"
cp target/release/sage-browser-worker "$contents/Helpers/sage-browser-worker"
cp target/release/sage-sandbox-worker "$contents/Helpers/sage-sandbox-worker"
cp target/release/sage-privileged-helper "$contents/Helpers/sage-privileged-helper"
cp apps/macos/Info.plist "$contents/Info.plist"
cp assets/icon.icns "$contents/Resources/sage.icns"
cp assets/icon-source-bg.png "$contents/Resources/sage-logo.png"

signing_identity=${SAGE_MACOS_SIGN_IDENTITY:--}
sign_binary() {
  binary=$1
  if [ "$signing_identity" = "-" ]; then
    codesign --force --sign - "$binary"
  else
    codesign --force --timestamp --options runtime --sign "$signing_identity" "$binary"
  fi
}

for helper in sage-core sage-browser-worker sage-sandbox-worker sage-privileged-helper; do
  sign_binary "$contents/Helpers/$helper"
done
if [ "$signing_identity" = "-" ]; then
  codesign --force --sign - --entitlements apps/macos/Sage.entitlements "$application"
else
  codesign \
    --force \
    --timestamp \
    --options runtime \
    --entitlements apps/macos/Sage.entitlements \
    --sign "$signing_identity" \
    "$application"
fi
codesign --verify --deep --strict --verbose=2 "$application"

mkdir -p "$(dirname "$disk_image")"
rm -f "$disk_image"
hdiutil create \
  -volname "Sage" \
  -srcfolder "$application" \
  -ov \
  -format UDZO \
  "$disk_image"
hdiutil verify "$disk_image"

if [ -n "${SAGE_NOTARY_PROFILE:-}" ]; then
  if [ "$signing_identity" = "-" ]; then
    echo "SAGE_NOTARY_PROFILE requires a Developer ID signing identity" >&2
    exit 1
  fi
  xcrun notarytool submit "$disk_image" --keychain-profile "$SAGE_NOTARY_PROFILE" --wait
  xcrun stapler staple "$disk_image"
  xcrun stapler validate "$disk_image"
fi

echo "$disk_image"
