#!/usr/bin/env bash
# MVTec AD 다운로드 + 무결성 검증 + 압축 해제.
# 공식 배포 링크는 신청서 뒤에 있어 자동화가 어렵다. 같은 아카이브를 올린 Hugging Face 미러를
# 쓰고, SHA-256을 anomalib이 공식 아카이브에 쓰는 값과 대조해 원본과 동일함을 확인한다.
# 라이선스: CC BY-NC-SA 4.0 (비상업). 저장소에는 데이터를 커밋하지 않는다.
set -euo pipefail

ROOT="${1:-$HOME/data/mvtec_ad}"
URL="https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz"
SHA256="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
ARCHIVE="$ROOT/_archive/mvtec_anomaly_detection.tar.xz"

mkdir -p "$ROOT/_archive"

if [[ -f "$ROOT/zipper/test/good/000.png" ]]; then
    echo "already extracted: $ROOT"
    exit 0
fi

if [[ ! -f "$ARCHIVE" ]] || ! echo "$SHA256  $ARCHIVE" | sha256sum -c --quiet - 2>/dev/null; then
    echo "downloading -> $ARCHIVE"
    curl -L --fail --retry 5 -C - -o "$ARCHIVE" "$URL"
    echo "$SHA256  $ARCHIVE" | sha256sum -c -
fi

echo "extracting -> $ROOT"
tar -xJf "$ARCHIVE" -C "$ROOT"
echo "done"
