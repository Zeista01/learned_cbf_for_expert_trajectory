#!/bin/bash
# Clearance sweep: re-run the full generalization protocol at widening
# guaranteed passages, to find where ours reliably REACHES the target
# (surgery needs the tool to arrive at the tumour), while safety stays intact.
set -e
cd "$(dirname "$0")/../.."
OUT="baselines/clearance_sweep"
for MM in 20 25 30; do
  CLR=$(python3 -c "print($MM/1000)")
  echo "=================== clearance ${MM}mm ==================="
  venv/bin/python -u baselines/run_full.py \
      --clearance "$CLR" --per 6 --starts 10 --workers 11 \
      --outdir "$OUT" --out "clr${MM}mm" \
      2>&1 | grep -vE "Polishing|Axes3D|warnings.warn|Loaded|UserWarning|self._centers|torch.tensor"
done
echo "ALL CLEARANCES DONE"
