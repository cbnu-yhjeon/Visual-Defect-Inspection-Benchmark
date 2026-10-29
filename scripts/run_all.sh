#!/usr/bin/env bash
# 전체 실험. 고전 기법(CPU 위주)과 GPU 방법을 두 줄로 나눠 병렬 실행한다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs cache/fonts
# 그림의 한글 라벨용 폰트 (OFL)
[[ -f cache/fonts/NanumGothic-Regular.ttf ]] || curl -sfL -o cache/fonts/NanumGothic-Regular.ttf \
    https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf || true
( $PY -m src.run --method A1 && $PY -m src.run --method A2 ) > logs/classical.log 2>&1 &
( $PY -m src.run --method C2 && $PY -m src.run --method B && $PY -m src.run --method C1 ) > logs/deep.log 2>&1 &
wait
$PY -m scripts.make_report
