#!/bin/bash
# One-time setup of ONE TPU VM (run it on every worker of the slice, e.g. via
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "bash ~/LookAheadOPD/scripts/tpu/setup_node.sh"
# after the repo is on each worker). Idempotent: safe to re-run.
#
# Produces: ~/venv-vllm-tpu (exact pinned env, see requirements-tpu.txt), the TPU v4 kernel
# patch applied, ALFWorld games in ~/alf-data with the task files of data/alfworld pointing at them, and the HF
# models ($MODELS, default the 4B student; MODELS="Qwen/Qwen3-1.7B Qwen/Qwen3-8B" for the small pair).
# WebShop's code, catalog and search index are installed separately (README.md).
set -euo pipefail

REPO=${REPO:-$HOME/LookAheadOPD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
ALF=${ALF:-$HOME/alf-data}
# Models for the Hugging Face cache on this VM (student + single-chip teachers). A large
# multi-chip teacher is better kept on tmpfs: see TEACHER_MODEL in start_cluster.sh.
MODELS=${MODELS:-"Qwen/Qwen3-4B"}
HERE=$REPO/scripts/tpu

# 1. uv + Python 3.12.14 + venv
command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
[ -x "$VENV/bin/python" ] || { "$UV" python install 3.12.14; "$UV" venv --python 3.12.14 "$VENV"; }

# 2. Exact package set. --no-deps on purpose: the tested env uses qwix 0.1.8 (needed by
#    google-tunix) although tpu-inference pins 0.1.2, so a normal resolve would fail.
VIRTUAL_ENV=$VENV "$UV" pip install --no-deps -r "$HERE/requirements-tpu.txt"
VIRTUAL_ENV=$VENV "$UV" pip install --no-deps -r "$HERE/requirements-envs.txt"  # ScienceWorld, WebShop
VIRTUAL_ENV=$VENV "$UV" pip install --no-deps -e "$REPO"

# 3. vLLM-TPU ragged-paged-attention v3 kernel: add TPU v4 tile sizes (v4 has 16 MiB VMEM).
SP=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
K=$SP/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
if ! grep -q "case 4:  # added locally" "$K"; then
    cp "$K" "$K.orig"
    patch -p1 -d "$SP" < "$HERE/tpu_inference_rpa_v3_tpu_v4.patch"
fi

# 4. ALFWorld games + the task files of data/alfworld (game paths relative to the ALFWorld data root) with
#    absolute game_file paths, in $ALF/lookahead_opd_tasks (the configs read train.jsonl from there)
[ -d "$ALF/json_2.1.1/valid_unseen" ] || "$VENV/bin/alfworld-download" --data-dir "$ALF"
mkdir -p "$ALF/lookahead_opd_tasks"
for f in train test test_unseen; do
    sed "s#\"game_file\": \"#\"game_file\": \"$ALF/#" "$REPO/data/alfworld/$f.jsonl" > "$ALF/lookahead_opd_tasks/$f.jsonl"
done
"$VENV/bin/python" - "$ALF" <<'EOF'
import json, os, sys
alf = sys.argv[1]
for f in ["train", "test", "test_unseen"]:
    rows = [json.loads(l) for l in open(f"{alf}/lookahead_opd_tasks/{f}.jsonl")]
    missing = sum(not os.path.exists(r["game_file"]) for r in rows)
    print(f"{f}: {len(rows)} tasks, {missing} missing game files")
    assert missing == 0, "ALFWorld download incomplete or task paths wrong"
EOF

# 5. Models (Hugging Face cache)
for m in $MODELS; do "$VENV/bin/hf" download "$m" >/dev/null; echo "downloaded $m"; done

# 6. Scratch dirs used by the configs / Ray
mkdir -p /dev/shm/tmp /dev/shm/ray_tmp /dev/shm/lookahead_opd_sync /dev/shm/lookahead_opd_buffers "$REPO/checkpoints"
echo "setup_node: done on $(hostname)"
