#!/usr/bin/env bash
# Download ONNX Runtime for Jetson (linux aarch64) into deploy/thirdparty/.
set -euo pipefail

ONNX_VERSION="1.22.0"
ARCHIVE="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ARCHIVE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRDPARTY_DIR="${REPO_ROOT}/deploy/thirdparty"
DEST_DIR="${THIRDPARTY_DIR}/onnxruntime-linux-aarch64-${ONNX_VERSION}"

if [[ -f "${DEST_DIR}/include/onnxruntime_cxx_api.h" ]]; then
  echo "Already installed: ${DEST_DIR}"
  exit 0
fi

mkdir -p "${THIRDPARTY_DIR}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Downloading ${ARCHIVE}..."
curl -L -o "${TMP_DIR}/${ARCHIVE}" "${URL}"

echo "Extracting to ${THIRDPARTY_DIR}..."
tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${THIRDPARTY_DIR}"

if [[ ! -f "${DEST_DIR}/lib/libonnxruntime.so.1.22.0" ]]; then
  echo "Unexpected archive layout. Expected: ${DEST_DIR}" >&2
  exit 1
fi

echo "Done: ${DEST_DIR}"

