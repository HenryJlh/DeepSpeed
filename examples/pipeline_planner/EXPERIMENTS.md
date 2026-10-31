# 在其他集群上跑流水线规划器（给合作者的说明）

这个分支 `x-moe-4d-v2-all` 只用来跑实验，不用来提 PR。它包含：
- 规划器本体，也就是分支 `x-moe-4d-v2`，对应 PR-A 和 PR-B；
- 4 个修复：最后一个 stage 释放输出、`seed_layers`、ZeRO-1 范数、AutoEP 流水线；
- 本目录的示例。

## 1. 安装

```bash
git clone -b x-moe-4d-v2-all https://github.com/HenryJlh/DeepSpeed.git
cd DeepSpeed
pip install -e .        # 或者不安装：export PYTHONPATH=$PWD
python -c "import deepspeed; print(deepspeed.__file__)"   # 确认导入的是这个目录，而不是环境里原有的 DeepSpeed
```

只需要 PyTorch 和 DeepSpeed 本身的依赖，不需要 Megatron 或 X-MoE。

## 2. 一键运行（推荐）

`run_experiment.sh` 会依次完成 profile、规划、按 plan 训练、用 DeepSpeed 默认切分跑对照、逐个 stage 对比显存。所有设置都通过环境变量给出，完整说明见脚本开头的注释。

```bash
cd examples/pipeline_planner
bash run_experiment.sh                                              # 默认：MoE，单节点 4 卡，PP4
MODEL=dense PP=4 bash run_experiment.sh                             # dense 7.5B
MODEL=moe PP=2 EP=2 bash run_experiment.sh                          # 4 卡：PP2 × DP2 × EP2
NUM_NODES=2 GPUS_PER_NODE=8 HOSTFILE=hosts PP=4 EP=4 bash run_experiment.sh    # 2 节点 16 卡：PP4 × DP4 × EP4
ALLOC=expandable_segments:True bash run_experiment.sh               # NVIDIA 上减少显存碎片（AMD 上用 max_split_size_mb:512）
MODEL=custom MODEL_ARGS="--layers 48 --hidden 4096 --seq 8192 --experts 16 --ffn 8192" bash run_experiment.sh
```

脚本会自动完成这些事：
- 由总卡数和 PP 算出 DP，由 MBS、GAS、DP 算出 global batch；
- 生成对应的 `ds_config.json`；
- 从 0 号卡读出每卡显存，也可以用 `MEM_GB` 指定。

结果都在 `runs/<模型>_pp<PP>_dp<DP>_ep<EP>/` 下：
- `summary.txt`：汇总，包括规划器的输出、各次训练每个 stage 的步时和显存、预测与实测的对比；
- `costs.json`、`plan.json`；
- 各步骤的完整日志。

把整个目录发回来即可。

多节点时，脚本本身在其中一个节点上运行；profile 只用这个节点上的 EP 张卡，训练通过 `HOSTFILE` 用 DeepSpeed launcher 拉起所有节点。如果集群只能用 srun 启动，按下面第 3 节的单步命令自己替换启动方式。

## 3. 单步运行

以下命令都在 `examples/pipeline_planner/` 目录下执行。`$M` 是模型参数，见 `model.py` 的 `add_model_args`，例如：

```bash
M="--layers 32 --hidden 2048 --seq 8192 --vocab 128000 --experts 8 --ffn 4096"
```

1. **profile**：测 Block、Embed、Head（含 loss）各自的开销，写出 `costs.json`。
   ```bash
   deepspeed --num_gpus <EP> profile_layers.py $M --dp <DP> --ep <EP> -o costs.json
   ```
   - `--dp`、`--ep` 填训练时的值。状态显存按 2 + 2 + 12/d 计算，step 显存按 6/d 计算：dense 参数的 d 取 DP，专家参数的 d 取 DP/EP。
   - MoE 模型要用 EP 个进程来 profile（`--num_gpus <EP>`），这样 all-to-all 的时间才会算进去。dense 模型用 1 张卡即可。
2. **规划**
   ```bash
   python -m deepspeed.runtime.pipe.planner --costs costs.json --stages <PP> --memory-gb <每卡显存 GiB> \
       --global-batch-size <GBS> --dp <DP> --micro-batch-sizes 1 --partition-layer-type Block -o plan.json
   ```
   这里 `GBS = micro-batch 大小 × gradient_accumulation_steps × DP`，要和 `ds_config.json` 一致。
3. **训练**：先按 plan 跑，再用 DeepSpeed 默认切分加全重算跑一次作对照。
   ```bash
   deepspeed --hostfile hosts --num_nodes <N> --num_gpus <每节点卡数> train.py $M --ep <EP> \
       --stages <PP> --partition profile:plan.json --deepspeed_config ds_config.json > train_plan.log 2>&1
   deepspeed ... train.py $M --ep <EP> --stages <PP> --partition parameters --recompute-all \
       --deepspeed_config ds_config.json > train_base.log 2>&1
   python compare.py plan.json train_plan.log
   ```
   - 总卡数 = PP × DP，EP 必须能整除 DP。
   - `ds_config.json` 里的 `gradient_accumulation_steps` 就是每步的 micro-batch 数；和 plan 不一致时 engine 会打警告。

## 4. 注意事项

- **显存碎片**：如果预测没超、实际却 OOM，报错里 "reserved but unallocated" 有好几 GiB，就是碎片导致的。
  - NVIDIA：设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（新版 PyTorch 用 `PYTORCH_ALLOC_CONF`）。
  - AMD（ROCm）：`expandable_segments` 不生效，改设 `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512`。
  - 然后再调大 `--headroom-gb`（默认是卡显存的 10%）。
- **适用范围**：显存模型只对 bf16 + ZeRO-1 + Adam 成立。示例的 `ds_config.json` 就是这个配置。
- **每个 stage 的输出**：`train.py` 训练 2 步热身加 5 步计时，每个 stage 打印一行：平均步时、峰值 allocated 和 reserved 显存、loss。

## 5. 请发回的结果

用 `run_experiment.sh` 时，把 `runs/` 下对应的整个目录发回来，再附上 GPU 型号、节点数和 PyTorch / CUDA 或 ROCm 版本即可。单步运行时请发回：

1. 硬件和软件：GPU 型号和显存、节点数、PyTorch / CUDA 或 ROCm 版本，以及 `deepspeed.__file__` 那一行的输出；
2. `costs.json`、`plan.json`、规划器的完整输出（单步运行第 2 步打印的内容）；
3. `train_plan.log`、`train_base.log`（至少包含每个 stage 的 `stage N:` 那几行）和 `compare.py` 的输出；
4. 如果出现 OOM：报错里 "Tried to allocate ... reserved but unallocated" 的那一段。

## 6. 在 4×MI210 上已有的结果，供对照

| 模型 | plan | `parameters` 不重算 | `parameters` 全重算 | 显存预测误差 |
|---|---|---|---|---|
| dense 7.5B，seq 4096，PP4 × DP1 | 9.96 s | 10.18 s | 12.72 s | −0.4 ~ −1.6 GiB |
| MoE，8 专家，seq 8192，PP4 × DP1 | 9.83 s | 10.40 s | 13.08 s | −1.5 ~ −1.6 GiB |
| MoE，8 专家，seq 8192，PP2 × DP2 × EP2 | 19.08 s | 未跑 | 26.33 s | −1.3 ~ −1.6 GiB |

最后一行是用 `MODEL=moe PP=2 EP=2 bash run_experiment.sh` 跑出来的，验证了公式里按 1/d 切分的部分（状态的 12/d，以及专家参数按 dp/ep 切分）。

**还需要补的数据**：更大的 DP 和 EP、多节点，以及 NVIDIA GPU。另外，step 阶段的 6/d 目前只在 DP1 下逼近显存上限时验证过，在 DP 更大时它不会是峰值。
