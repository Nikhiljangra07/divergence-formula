#!/usr/bin/env bash
# Bigger v2 pilot — queued to run AFTER the Constellax loop exits (avoid OpenRouter DSV4 contention).
# Safe: waits for the loop, backs up the old pilot, hard cost caps on both passes.
set -u
cd "$(dirname "$0")"            # corpus_run/  (config.py + dav.py importable)

echo "[queue] waiting for Constellax autonomous loop to finish before starting (avoid DSV4 rate contention)..."
while pgrep -f run_autonomous_pipeline >/dev/null 2>&1; do sleep 30; done
echo "[queue] Constellax loop done at $(date '+%H:%M:%S') — starting bigger pilot."

# 1) preserve the existing n=12 pilot (WORKING_PAPER §8 cites it) before gen_v2 overwrites corpus_v2/
if [ -d corpus_v2 ]; then
  BK="corpus_v2_pilot_n12_$(date '+%Y%m%d-%H%M%S')"
  cp -R corpus_v2 "$BK"
  echo "[backup] old pilot -> $BK"
fi

# 2) bigger single-arm pass (target 300 -> ~461 planned candidates -> ~90 passers at the pilot's real yield)
echo "[gen_v2] starting target=300, cap \$4 ..."
V2_USD_CAP=4.0 python3 gen_v2.py --target 300
GEN_RC=$?
echo "[gen_v2] exit=$GEN_RC"

# 3) paired no-consequence B pass over the new passers -> clean A/B at large n
if [ "$GEN_RC" -eq 0 ] && [ -s corpus_v2/passers.jsonl ]; then
  echo "[gen_v2_b] starting paired B pass, cap \$2 ..."
  V2B_USD_CAP=2.0 python3 gen_v2_b.py
  echo "[gen_v2_b] exit=$?"
else
  echo "[gen_v2_b] SKIPPED (gen_v2 failed or no passers)"
fi

echo "[done] bigger pilot finished at $(date '+%H:%M:%S')"
echo "=== summary.json ==="; cat corpus_v2/summary.json 2>/dev/null
echo "=== ab_compare.json ==="; cat corpus_v2/ab_compare.json 2>/dev/null
