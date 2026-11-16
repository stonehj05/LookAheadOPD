#!/bin/bash
# LookAheadOPD end to end, scienceworld, Qwen3 30B -> 4B, on worker 0 of the TPU slice:
# start the cluster, train with configs/scienceworld/lookahead_opd_qwen3_30b_to_4b.yaml, then evaluate every saved checkpoint (avg@4, all chips).
#
#   bash scripts/run_scienceworld_qwen3_30b_to_4b.sh
#
# Run it inside tmux / screen or with `setsid nohup ... &`: it takes as long as the training plus the evals.
# Logs and results, with <run> = checkpoints/<project>/<name>:
#   <run>.train.log            the training console
#   <run>/log/, <run>/monitor/ trainer / explorer logs, TensorBoard
#   <run>.eval.log             one avg@4 summary per checkpoint (details: checkpoints/eval_results/)
# SKIP_CLUSTER=1 uses a Ray cluster that is already running; SKIP_TRAIN=1 only evaluates the run's checkpoints.
set -uo pipefail
CONFIG=configs/scienceworld/lookahead_opd_qwen3_30b_to_4b.yaml
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HOME/venv-vllm-tpu}
SSH="ssh -i ${SSH_KEY:-$HOME/.ssh/google_compute_engine} -o BatchMode=yes"
export REPO VENV PATH=$VENV/bin:$PATH
cd "$REPO"
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 1; }

ENV=$(basename "$(dirname "$CONFIG")")
case $ENV in
    alfworld) EVAL=scripts/tpu/eval/evaluate_checkpoint_avg4.sh ;;
    scienceworld) EVAL=scripts/tpu/eval/evaluate_sciworld_avg4.sh ;;
    webshop) EVAL=scripts/tpu/eval/evaluate_webshop_avg4.sh ;;
    *) echo "config must be under configs/{alfworld,scienceworld,webshop}/" >&2; exit 1 ;;
esac
PROJECT=$(sed -n 's/^project: *"\(.*\)"/\1/p' "$CONFIG")
NAME=$(sed -n 's/^name: *"\(.*\)"/\1/p' "$CONFIG")
RUN=$REPO/checkpoints/$PROJECT/$NAME
mkdir -p "$REPO/checkpoints/$PROJECT"
say() { echo "[$(date -u +%F' '%T)] $*"; }

# Workers: $WORKER_IPS ("ip0 ip1 ..."), else the TPU-VM metadata, else this VM.
md() { curl -sf -m 5 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true; }
if [ -n "${WORKER_IPS:-}" ]; then read -r -a IPS <<< "$WORKER_IPS"
else mapfile -t IPS < <(md attributes/worker-network-endpoints | tr ',' '\n' | awk -F: 'NF>=3 {print $3}'); fi
[ "${#IPS[@]}" -gt 0 ] || IPS=(localhost)
on() { if [ "$1" = localhost ] || hostname -I | tr ' ' '\n' | grep -qx "$1"; then bash -c "$2"; else $SSH "$1" "$2"; fi; }

if [ -z "${SKIP_TRAIN:-}" ]; then
    if [ -z "${SKIP_CLUSTER:-}" ]; then
        # 30B -> 4B: two dedicated teacher workers, the teacher fetched to tmpfs; 8B -> 1.7B: default roles
        if grep -q "Qwen3-30B-A3B" "$CONFIG"; then
            WORKER_ROLES="trainer teacher teacher explorer" TEACHER_MODEL=Qwen/Qwen3-30B-A3B \
                bash scripts/tpu/start_cluster.sh || { say "start_cluster failed"; exit 1; }
        else
            bash scripts/tpu/start_cluster.sh || { say "start_cluster failed"; exit 1; }
        fi
    fi
    say "training $CONFIG -> $RUN (console: $RUN.train.log)"
    trinity run --config "$CONFIG" > "$RUN.train.log" 2>&1 < /dev/null
    rc=$?
    say "training finished (exit $rc)"
    [ "$rc" = 0 ] || { say "training failed; see $RUN.train.log"; exit 1; }
    # Free the chips for the eval servers: stop Ray everywhere, wait until no process holds a TPU chip.
    for ip in "${IPS[@]}"; do on "$ip" "$VENV/bin/ray stop --force > /dev/null 2>&1"; done
    for try in $(seq 1 60); do
        busy=0
        for ip in "${IPS[@]}"; do busy=$(( busy + $(on "$ip" 'lsof -t /dev/accel* 2>/dev/null | wc -l') )); done
        [ "$busy" = 0 ] && break
        sleep 10
    done
    [ "$busy" = 0 ] || { say "TPU chips still busy after training; evaluate later with SKIP_TRAIN=1"; exit 1; }
fi

mapfile -t STEPS < <(ls -d "$RUN"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)
[ "${#STEPS[@]}" -gt 0 ] || { say "no checkpoints in $RUN"; exit 1; }
SUMMARY=$RUN.eval.log
say "evaluating steps ${STEPS[*]} of $RUN (summary: $SUMMARY)" | tee -a "$SUMMARY"
for step in "${STEPS[@]}"; do
    ckpt=$RUN/global_step_$step/actor/huggingface
    [ -f "$ckpt/model.safetensors" ] || { say "step $step: no checkpoint, skipped" | tee -a "$SUMMARY"; continue; }
    log=$RUN.eval_step$step.log
    bash "$EVAL" "$ckpt" "${NAME}_step$step" > "$log" 2>&1
    say "step $step done (details: $log)" | tee -a "$SUMMARY"
    grep -E "avg@" "$log" | sed "s/^/    step $step  /" | tee -a "$SUMMARY"
done
say "ALL DONE" | tee -a "$SUMMARY"
