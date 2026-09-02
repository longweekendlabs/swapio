#!/usr/bin/env bash
# Build an AppImage from an existing dist/swapio bundle.
#
#   ./packaging/build_rpm.sh --gpu      # produces dist/swapio/
#   ./packaging/build_appimage.sh       # packages it
#
# appimagetool is fetched once into packaging/.tools and reused. It is run with
# --appimage-extract-and-run so no FUSE is needed, which matters inside CI.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

[[ -x dist/swapio/swapio ]] || {
  echo "No dist/swapio/swapio. Build the bundle first." >&2
  exit 1
}

version="$(.venv/bin/python -c 'from version import VERSION; print(VERSION)')"
case "$(uname -m)" in
  x86_64) architecture="x86_64" ;;
  aarch64 | arm64) architecture="aarch64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 2 ;;
esac
appdir="$project_dir/appimage-build/Swapio.AppDir"
tools="$project_dir/packaging/.tools"
output_dir="$project_dir/dist"

echo "=== Swapio v${version} - AppImage (${architecture}) ==="

rm -rf "$project_dir/appimage-build"
mkdir -p "$appdir/usr/bin" \
         "$appdir/usr/share/applications" \
         "$appdir/usr/share/icons/hicolor/scalable/apps"

cp -a dist/swapio/. "$appdir/usr/bin/"
cp packaging/swapio.desktop "$appdir/usr/share/applications/swapio.desktop"
cp packaging/swapio.desktop "$appdir/swapio.desktop"
cp assets/swapio.svg "$appdir/usr/share/icons/hicolor/scalable/apps/swapio.svg"
cp assets/swapio.svg "$appdir/swapio.svg"
cp assets/swapio.svg "$appdir/.DirIcon"

# An AppImage is relocatable, so the launcher cannot point at /opt.
cat > "$appdir/AppRun" <<'EOF'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/usr/bin/swapio" "$@"
EOF
chmod 755 "$appdir/AppRun"

mkdir -p "$tools"
appimagetool="$tools/appimagetool-${architecture}.AppImage"
if [[ ! -x "$appimagetool" ]]; then
  echo "Fetching appimagetool for ${architecture}..."
  curl -fsSL -o "$appimagetool" \
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${architecture}.AppImage"
  chmod 755 "$appimagetool"
fi

mkdir -p "$output_dir"
final="$output_dir/Swapio-${version}-${architecture}.AppImage"
rm -f "$final"
ARCH="$architecture" "$appimagetool" --appimage-extract-and-run --no-appstream "$appdir" "$final"
chmod 755 "$final"

echo
echo "AppImage ready: $final"
echo "Run it with: chmod +x '$final' && '$final'"
