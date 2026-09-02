#!/usr/bin/env bash
# Build a Debian/Ubuntu package from an existing dist/swapio bundle.
#
#   ./packaging/build_rpm.sh --gpu    # produces dist/swapio/
#   ./packaging/build_deb.sh          # packages it
#
# Like the RPM, this declares no dependencies. The PyInstaller bundle carries
# its own Qt and ONNX Runtime, and a wrong Depends line would block the install
# outright rather than fail politely at launch.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

[[ -x dist/swapio/swapio ]] || {
  echo "No dist/swapio/swapio. Build the bundle first." >&2
  exit 1
}

version="$(.venv/bin/python -c 'from version import VERSION; print(VERSION)')"
case "$(uname -m)" in
  x86_64) architecture="amd64" ;;
  aarch64 | arm64) architecture="arm64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 2 ;;
esac
stage="$project_dir/deb-build/stage"
output_dir="$project_dir/dist"

echo "=== Swapio v${version} - DEB (${architecture}) ==="

rm -rf "$project_dir/deb-build"
mkdir -p "$stage/DEBIAN" \
         "$stage/opt/swapio" \
         "$stage/usr/bin" \
         "$stage/usr/share/applications" \
         "$stage/usr/share/icons/hicolor/scalable/apps"

cp -a dist/swapio/. "$stage/opt/swapio/"
cp packaging/swapio.desktop "$stage/usr/share/applications/swapio.desktop"
cp assets/swapio.svg "$stage/usr/share/icons/hicolor/scalable/apps/swapio.svg"
cp packaging/swapio-launcher "$stage/usr/bin/swapio"
chmod 755 "$stage/usr/bin/swapio"

installed_kb="$(du -sk "$stage" | cut -f1)"
cat > "$stage/DEBIAN/control" <<EOF
Package: swapio
Version: ${version}
Architecture: ${architecture}
Maintainer: Long Weekend Labs <2272935+longweekendlabs@users.noreply.github.com>
Section: graphics
Priority: optional
Installed-Size: ${installed_kb}
Homepage: https://github.com/longweekendlabs/swapio
Description: Private offline batch face swapping for still photos
 Swapio replaces one source face across a batch of destination photos.
 Processing runs locally, original files are never changed, and the
 separately licensed pretrained models are downloaded and verified on
 first use.
EOF

cat > "$stage/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
update-desktop-database >/dev/null 2>&1 || true
gtk-update-icon-cache /usr/share/icons/hicolor >/dev/null 2>&1 || true
EOF
cp "$stage/DEBIAN/postinst" "$stage/DEBIAN/postrm"
chmod 755 "$stage/DEBIAN/postinst" "$stage/DEBIAN/postrm"

mkdir -p "$output_dir"
final="$output_dir/swapio_${version}_${architecture}.deb"
dpkg-deb --build --root-owner-group -Zxz "$stage" "$final"

echo
echo "DEB ready: $final"
echo "Install with: sudo apt install '$final'"
