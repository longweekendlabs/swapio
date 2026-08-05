#!/usr/bin/env bash
# Build a Fedora/RHEL-family RPM from the current Swapio working tree.
#
# Public/model-less package (recommended for releases):
#   ./packaging/build_rpm.sh
#
# Local test package containing already-installed models:
#   ./packaging/build_rpm.sh --bundle-models
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

bundle_models=0
reuse_bundle=0
for argument in "$@"; do
  case "$argument" in
    --bundle-models) bundle_models=1 ;;
    --reuse-bundle) reuse_bundle=1 ;;
    *) echo "Usage: $0 [--bundle-models] [--reuse-bundle]" >&2; exit 2 ;;
  esac
done
if [[ $bundle_models -eq 1 && $reuse_bundle -eq 1 ]]; then
  echo "--reuse-bundle cannot be combined with --bundle-models" >&2
  exit 2
fi

version="$(.venv/bin/python -c 'from version import VERSION; print(VERSION)')"
release="2"
architecture="$(uname -m)"
top="$project_dir/rpm-build"
stage="$top/stage"
output_dir="$project_dir/dist"

echo "=== Swapio v${version} — RPM (${architecture}) ==="

if [[ $reuse_bundle -eq 1 ]]; then
  echo "Reusing the existing verified dist/swapio bundle."
elif [[ $bundle_models -eq 1 ]]; then
  if [[ ! -x .venv/bin/pyinstaller ]]; then
    .venv/bin/pip install pyinstaller
  fi
  .venv/bin/python -c 'import core; missing=core.missing_models(); assert not missing, "Missing models: " + ", ".join(missing)'
  SWAPIO_BUNDLE_MODELS=1 .venv/bin/pyinstaller --noconfirm --clean packaging/swapio.spec
else
  build_venv="$project_dir/.package-venv"
  if [[ ! -x "$build_venv/bin/python" ]]; then
    .venv/bin/python -m venv "$build_venv"
  fi
  "$build_venv/bin/pip" install -r requirements.txt pyinstaller
  "$build_venv/bin/pyinstaller" --noconfirm --clean packaging/swapio.spec
fi

[[ -x dist/swapio/swapio ]] || { echo "PyInstaller produced no dist/swapio/swapio" >&2; exit 1; }

rm -rf "$top"
mkdir -p "$top"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
mkdir -p "$stage/opt/swapio" \
         "$stage/usr/bin" \
         "$stage/usr/share/applications" \
         "$stage/usr/share/icons/hicolor/scalable/apps"

cp -a dist/swapio/. "$stage/opt/swapio/"
cp packaging/swapio.desktop "$stage/usr/share/applications/swapio.desktop"
cp assets/swapio.svg "$stage/usr/share/icons/hicolor/scalable/apps/swapio.svg"
cp packaging/swapio-launcher "$stage/usr/bin/swapio"
chmod 755 "$stage/usr/bin/swapio"

cat > "$top/SPECS/swapio.spec" <<EOF
%global __os_install_post %{nil}
%global debug_package %{nil}
%global _binary_payload w3.zstdio
AutoReqProv: no

Name:           swapio
Version:        ${version}
Release:        ${release}%{?dist}
Summary:        Private offline batch face swapping for still photos
License:        LicenseRef-Proprietary
URL:            https://github.com/longweekendlabs/swapio
BuildArch:      ${architecture}

%description
Swapio replaces one source face across a batch of destination photos. Processing
runs locally, original files are never changed, and public packages download and
verify the separately licensed pretrained models on first use.

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}
cp -a ${stage}/. %{buildroot}/

%files
/opt/swapio
/usr/bin/swapio
/usr/share/applications/swapio.desktop
/usr/share/icons/hicolor/scalable/apps/swapio.svg

%post
/usr/bin/update-desktop-database &>/dev/null || :
/bin/touch --no-create /usr/share/icons/hicolor &>/dev/null || :

%postun
/usr/bin/update-desktop-database &>/dev/null || :
/usr/bin/gtk-update-icon-cache /usr/share/icons/hicolor &>/dev/null || :
EOF

rpmbuild --define "_topdir $top" -bb "$top/SPECS/swapio.spec"
mkdir -p "$output_dir"
rpm_path="$(find "$top/RPMS" -name '*.rpm' -print -quit)"
suffix=""
if [[ $bundle_models -eq 1 ]]; then
  suffix="-with-models"
fi
final="$output_dir/swapio-${version}-${release}-${architecture}${suffix}.rpm"
cp "$rpm_path" "$final"

echo
echo "RPM ready: $final"
echo "Install with: sudo dnf install '$final'"
