# LookAheadOPD

On-policy distillation (OPD) for multi-turn agents in which the teacher decides, turn by turn, whether and how
strongly its signal is applied: by whether it agrees with the student's action, and by looking ahead at what
the student does next and at what the teacher itself would do from the same position.

This repository contains LookAheadOPD's code and the configs of its ALFWorld, ScienceWorld and WebShop runs
(Qwen3-30B-A3B → Qwen3-4B and Qwen3-8B → Qwen3-1.7B), on Cloud TPU, built on
[Trinity-RFT](https://github.com/modelscope/Trinity-RFT) and the [TCOD](https://github.com/kokolerk/TCOD)
codebase.

## Repository layout

| Path | What |
|---|---|
| `trinity/common/workflows/envs/lookahead_opd/gate.py` | the method: prompts, answer parsing, weights, per-turn payload, look-ahead Ray task |
| `trinity/common/workflows/envs/lookahead_opd/workflow.py` | explorer side: the student's episode, a payload on every turn |
| `trinity/common/workflows/envs/lookahead_opd/{alfworld,scienceworld,webshop}.py` | the environments: rollout and teacher look-ahead (`LookAheadOPD_<env>_workflow`) |
| `trinity/common/workflows/envs/lookahead_opd/*_utils.py`, `webshop_env_server.py` | prompts and game helpers; shared WebShop servers |
| `trinity/trainer/tunix/teacher_gate.py` | trainer side: the gate for a sampled batch, teacher scoring, gate metrics |
| `trinity/trainer/tunix_trainer.py`, `trinity/trainer/tunix/` | the TPU (JAX/Tunix) trainer |
| `trinity/common/models/vllm_tpu_*.py`, `tpu_env.py` | vLLM-TPU engines for the student and teacher |
| `configs/<env>/lookahead_opd_qwen3_{30b_to_4b,8b_to_1.7b}.yaml` | the six training configs |
| `data/` | task splits: ALFWorld and ScienceWorld (TCOD's), WebShop sessions (TCOD's) |
| `scripts/tpu/` | per-VM setup, cluster start |
| `scripts/tpu/eval/` | avg@4 evaluation on all chips; `client/` holds the eval clients |

## Hardware

| Config | Chips | Layout |
|---|---|---|
| `*_qwen3_30b_to_4b.yaml` | 16 | worker 0: trainer (4); workers 1–2: one Qwen3-30B-A3B teacher each (TP 4); worker 3: 4 students |
| `alfworld/`, `webshop/` `*_qwen3_8b_to_1.7b.yaml` | 8 | worker 0: trainer (4); worker 1: 2 students and 2 Qwen3-8B teachers (one chip each) |
| `scienceworld/*_qwen3_8b_to_1.7b.yaml` | 16 | worker 0: trainer (4); the other 12 chips: 2 students and 10 Qwen3-8B teachers |

## Setup

This setup is specific to Google Cloud TPU instances; GPU configs are in progress.

All commands run on worker 0 unless stated. The paths assume the repo at `~/LookAheadOPD` and the environment
at `~/venv-vllm-tpu` on every worker; override them with `REPO` and `VENV`.

**1. SSH between workers.** Do this once per slice. `gcloud` pushes your key the first time; afterwards the
scripts use plain ssh over internal IPs with `~/.ssh/google_compute_engine` (override with `SSH_KEY`).

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command true
```

**2. Code, environment, data and models on every worker.** Do this once per slice; it takes about 20–30 min.
Copy the repo to `~/LookAheadOPD` on every worker, then:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command \
  'MODELS="Qwen/Qwen3-4B Qwen/Qwen3-1.7B Qwen/Qwen3-8B" bash ~/LookAheadOPD/scripts/tpu/setup_node.sh'
```

`setup_node.sh` is idempotent. On each VM it:

1. Installs `uv`, Python 3.12.14 and the venv.
2. Installs the exact tested package set with `--no-deps`: `requirements-tpu.txt`, plus `requirements-envs.txt`
   for ScienceWorld and WebShop. It then installs the repo as an editable package. A normal dependency
   resolve fails, because `tpu-inference` pins `qwix==0.1.2` while `google-tunix` needs 0.1.8.
3. Applies `tpu_inference_rpa_v3_tpu_v4.patch`, which adds the TPU v4 tile sizes that vLLM-TPU's attention
   kernel lacks.
4. Downloads ALFWorld to `~/alf-data`. It writes `data/alfworld/*.jsonl`, with absolute `game_file` paths, to
   `~/alf-data/lookahead_opd_tasks/`, where the ALFWorld configs read them.
5. Downloads `$MODELS` to the Hugging Face cache. The 30B teacher is fetched to tmpfs by `start_cluster.sh`
   instead.

ScienceWorld also needs Java (`java -version`); its task files in `data/scienceworld/` use the game bundled
with the `scienceworld` package.

**3. WebShop**, only for the WebShop runs, on every worker. The runs use WebShop's full catalog (1.18M
products) and human goals: training sessions 0–4095 and test sessions 4096–4195 need it.

1. Clone [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop). Its original data links have
   expired, so we use a mirror: download `items_shuffle.json`, `items_ins_v2.json` and `items_human_ins.json`
   from [YWZBrandon/webshop-data](https://huggingface.co/datasets/YWZBrandon/webshop-data) on Hugging Face
   into WebShop's `data/` folder.
2. In `web_agent_site/utils.py`, point `DEFAULT_FILE_PATH` and `DEFAULT_ATTR_PATH` at the full files:
   `../data/items_shuffle.json` and `../data/items_ins_v2.json`.
3. Build the search index with `search_engine/run_indexing.sh`. The index is about 3.4 GB.
4. Make `web_agent_site` importable from the venv, for example with a `.pth` file in its `site-packages`
   that holds WebShop's directory. Also install spaCy's model: `python -m spacy download en_core_web_sm`.

Each WebShop server actor loads the catalog once, which takes about 20 GB of RAM and 2 minutes. Training starts
4 servers (`webshop_env_server.py`), and the student's games and the teacher's replays share them. Keeping the
catalog and index on tmpfs speeds up loading.

## Training

Each run has one script that runs the train and the evaluation; run it on worker 0:

```bash
cd ~/LookAheadOPD
bash scripts/run_alfworld_qwen3_30b_to_4b.sh
bash scripts/run_alfworld_qwen3_8b_to_1.7b.sh
bash scripts/run_scienceworld_qwen3_30b_to_4b.sh
bash scripts/run_scienceworld_qwen3_8b_to_1.7b.sh
bash scripts/run_webshop_qwen3_30b_to_4b.sh
bash scripts/run_webshop_qwen3_8b_to_1.7b.sh
```


Run output goes to `checkpoints/<project>/<name>/`:

| Output | Where |
|---|---|
| per-step trainer metrics | `log/trainer.log` |
| per-step rollout metrics | `log/explorer.log` |
| TensorBoard | `monitor/` |
| full checkpoints | `global_step_<N>/actor/huggingface` |


## Evaluation

The evaluation is avg@4 over all chips of the slice. Each job starts one vLLM-TPU server per chip and
plays the test games with the training rollout's prompts and the full conversation as memory. The settings
are temperature 0.4, max_tokens 4096, a 40,960-token context and thinking off.

## Acknowledgements and license

LookAheadOPD is built on [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) and the
[TCOD](https://github.com/kokolerk/TCOD) codebase: its environments' setups and task splits. The ALFWorld,
ScienceWorld and WebShop prompts are those of [DASH-OPD](https://github.com/Lucian1115/DASH-OPD)
(arXiv 2607.29078). WebShop is [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop). Apache 2.0
(`LICENSE`).
