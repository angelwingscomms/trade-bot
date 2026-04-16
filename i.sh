#!/usr/bin/env bash
set -euo pipefail

days=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      days="${2:?missing day count for -i}"
      shift 2
      ;;
    *)
      echo "usage: ./i.sh [-i days]" >&2
      exit 1
      ;;
  esac
done

python_bin="python"
if [[ -x "./env/bin/python" ]]; then
  python_bin="./env/bin/python"
fi

train_output="$("$python_bin" nn.py)"
printf '%s\n' "$train_output"
model_folder="$(printf '%s\n' "$train_output" | awk 'NF {line=$0} END {print line}')"
if [[ -z "$model_folder" ]]; then
  echo "nn.py did not print a model folder name" >&2
  exit 1
fi

date_range="$("$python_bin" - <<PY
from datetime import date, timedelta
days = int(${days})
end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=days - 1)
print(start_date.isoformat(), end_date.isoformat())
PY
)"
from_date="$(printf '%s' "$date_range" | awk '{print $1}')"
to_date="$(printf '%s' "$date_range" | awk '{print $2}')"

"$python_bin" test.py --revision "$model_folder" --from-date "$from_date" --to-date "$to_date"

