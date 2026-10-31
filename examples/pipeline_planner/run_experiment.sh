#!/bin/bash
# One experiment end to end: profile -> plan -> train with the plan -> (optional) baselines -> compare.
# Everything is set through environment variables; the defaults run the MoE model on one node of 4 GPUs.
#
#   bash run_experiment.sh                                   # defaults below
#   MODEL=dense PP=4 bash run_experiment.sh
#   MODEL=moe PP=2 EP=2 GPUS_PER_NODE=4 bash run_experiment.sh            # PP2 x DP2 x EP2 on 4 GPUs
#   NUM_NODES=2 HOSTFILE=hosts PP=4 EP=8 GPUS_PER_NODE=8 bash run_experiment.sh
#
# Settings (environment variables):
#   MODEL          dense | moe | custom (then MODEL_ARGS is used as is)                       [moe]
#   MODEL_ARGS     model arguments of model.py, e.g. "--layers 32 --hidden 2048 --experts 8 ..." [preset of MODEL]
#   GPUS_PER_NODE  GPUs per node                                                              [4]
#   NUM_NODES      nodes                                                                      [1]
#   HOSTFILE       DeepSpeed hostfile for more than one node                                  []
#   PP             pipeline stages; DP = NUM_NODES * GPUS_PER_NODE / PP                        [4]
#   EP             expert-parallel size (MoE only); must divide DP                            [1]
#   MBS            micro-batch size                                                           [1]
#   GAS            micro-batches per step (gradient accumulation steps)                       [16]
#   MEM_GB         device memory per GPU in GiB                                               [read from GPU 0]
#   HEADROOM_GB    memory left out of the plan                                                [planner default, 10 %]
#   BASELINES      DeepSpeed partitions to run for comparison: any of full, none              [full]
#   ALLOC          allocator setting against fragmentation, e.g. max_split_size_mb:512 (ROCm)
#                  or expandable_segments:True (CUDA)                                         []
#   OUT            output directory                                                           [runs/<MODEL>_pp<PP>_dp<DP>_ep<EP>]
set -eo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-moe}
case $MODEL in
  dense) PRESET="--layers 32 --hidden 4096 --seq 4096 --vocab 128000 --ffn 16384" ;;
  moe)   PRESET="--layers 32 --hidden 2048 --seq 8192 --vocab 128000 --experts 8 --ffn 4096" ;;
  custom) PRESET="" ;;
  *) echo "MODEL must be dense, moe or custom" >&2; exit 1 ;;
esac
MODEL_ARGS=${MODEL_ARGS:-$PRESET}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NUM_NODES=${NUM_NODES:-1}
PP=${PP:-4}
EP=${EP:-1}
MBS=${MBS:-1}
GAS=${GAS:-16}
BASELINES=${BASELINES:-full}
WORLD=$((NUM_NODES * GPUS_PER_NODE))
if [ $((WORLD % PP)) -ne 0 ]; then echo "PP=$PP must divide $WORLD GPUs" >&2; exit 1; fi
DP=$((WORLD / PP))
if [ $((DP % EP)) -ne 0 ]; then echo "EP=$EP must divide DP=$DP" >&2; exit 1; fi
GBS=$((MBS * GAS * DP))
OUT=${OUT:-runs/${MODEL}_pp${PP}_dp${DP}_ep${EP}}
mkdir -p "$OUT"
if [ -n "$ALLOC" ]; then
  export PYTORCH_ALLOC_CONF=$ALLOC PYTORCH_CUDA_ALLOC_CONF=$ALLOC PYTORCH_HIP_ALLOC_CONF=$ALLOC
fi
if [ -z "$MEM_GB" ]; then
  MEM_GB=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))")
fi
LAUNCH="deepspeed --num_nodes $NUM_NODES --num_gpus $GPUS_PER_NODE"
[ -n "$HOSTFILE" ] && LAUNCH="deepspeed --hostfile $HOSTFILE --num_nodes $NUM_NODES --num_gpus $GPUS_PER_NODE"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
log "deepspeed: $(python -c 'import deepspeed; print(deepspeed.__file__)')"
log "model: $MODEL_ARGS"
log "$WORLD GPUs = PP $PP x DP $DP, EP $EP; micro-batch $MBS x $GAS per step, global batch $GBS; ${MEM_GB} GiB per GPU"
log "allocator: ${ALLOC:-default}"

# DeepSpeed config: ds_config.json with this run's micro-batch size and count
python - "$MBS" "$GAS" "$OUT/ds_config.json" <<'EOF'
import json, sys
cfg = json.load(open("ds_config.json"))
cfg["train_micro_batch_size_per_gpu"], cfg["gradient_accumulation_steps"] = int(sys.argv[1]), int(sys.argv[2])
json.dump(cfg, open(sys.argv[3], "w"), indent=2)
EOF

# 1. Profile each kind of layer, on EP GPUs of this node so that the expert all-to-all is timed with it
log "1/4 profile on $EP GPU(s)"
deepspeed --num_gpus $EP profile_layers.py $MODEL_ARGS --dp $DP --ep $EP --micro-batch-size $MBS \
    -o "$OUT/costs.json" > "$OUT/profile.log" 2>&1 || { log "profiling failed, see $OUT/profile.log"; exit 1; }
grep -E "^(Block|Embed|Head) " "$OUT/profile.log" | tee -a "$OUT/summary.txt"

# 2. Plan
log "2/4 plan"
python -m deepspeed.runtime.pipe.planner --costs "$OUT/costs.json" --stages $PP --memory-gb $MEM_GB \
    ${HEADROOM_GB:+--headroom-gb $HEADROOM_GB} --global-batch-size $GBS --dp $DP --micro-batch-sizes $MBS \
    --partition-layer-type Block -o "$OUT/plan.json" 2>&1 | tee "$OUT/planner.log" | tee -a "$OUT/summary.txt" ||
    { log "no plan fits, see $OUT/planner.log"; exit 1; }

# 3. Train with the plan, then the baselines; a run that fails (e.g. out of memory) does not stop the others
results() { { grep -ohE "stage [0-9]+: (out of memory|[0-9]+ ms/step[^,]*, peak [^,]*, loss [0-9.]+)" "$1" || true; } | sort -u; }
train() {  # name partition [extra args]
  local name=$1 partition=$2; shift 2
  log "train: $name"
  set +e
  $LAUNCH train.py $MODEL_ARGS --ep $EP --stages $PP --partition "$partition" "$@" \
      --deepspeed_config "$OUT/ds_config.json" > "$OUT/train_$name.log" 2>&1
  local rc=$?
  set -e
  log "  exit $rc"; results "$OUT/train_$name.log" | tee -a "$OUT/summary.txt"
}
log "3/4 train"
train plan "profile:$OUT/plan.json"
for b in $BASELINES; do
  case $b in
    full) train parameters_full parameters --recompute-all ;;
    none) train parameters_none parameters ;;
  esac
done

# 4. Predicted vs measured memory per stage for the plan
log "4/4 compare"
python compare.py "$OUT/plan.json" "$OUT/train_plan.log" | tee -a "$OUT/summary.txt"
log "done: $OUT/summary.txt"
