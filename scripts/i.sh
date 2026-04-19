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
      echo "usage: ./scripts/i.sh [-i days]" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python_bin="python"
if [[ -x "$ROOT_DIR/env/bin/python" ]]; then
  python_bin="$ROOT_DIR/env/bin/python"
fi

train_output="$("$python_bin" "$SCRIPT_DIR/nn.py")"
printf '%s\n' "$train_output"
model_folder="$(printf '%s\n' "$train_output" | awk 'NF {line=$0} END {print line}')"
if [[ -z "$model_folder" ]]; then
  echo "nn.py did not print a model folder name" >&2
  exit 1
fi

date_range="$(
  "$python_bin" -c "
from datetime import date, timedelta
d = int('$days')
end = date.today() - timedelta(days=1)
start = end - timedelta(days=d - 1)
print(start.isoformat(), end.isoformat())
"
)"
from_date="$(printf '%s' "$date_range" | awk '{print $1}')"
to_date="$(printf '%s' "$date_range" | awk '{print $2}')"

"$python_bin" "$SCRIPT_DIR/test.py" --revision "$model_folder" --from-date "$from_date" --to-date "$to_date"
