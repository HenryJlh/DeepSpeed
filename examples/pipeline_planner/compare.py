# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compare a plan's predicted peak memory and step time per stage with what train.py measured.

    python compare.py plan.json train.log
"""

import json
import re
import sys

GiB = 2**30
LINE = re.compile(r"stage (\d+): (\d+) ms/step, peak ([\d.]+) GiB allocated / ([\d.]+) GiB reserved, loss ([\d.]+)")

plan = json.load(open(sys.argv[1]))
text = open(sys.argv[2]).read()
# ranks print concurrently, so several result lines can share one line of the log
measured = {int(m.group(1)): m.groups()[1:] for m in LINE.finditer(text)}
oom = sorted(set(re.findall(r"stage \d+: out of memory", text)))

print(f"plan: layers {plan['layers_per_stage']}, recomputed {plan['checkpoint_layers_per_stage']}, "
      f"micro-batch size {plan['micro_batch_size']} x {plan['micro_batches']}, "
      f"predicted step {plan['predicted_step_ms']:.0f} ms")
if oom:
    print("OUT OF MEMORY: " + ", ".join(oom))
print(f"{'stage':>5} {'predicted':>10} {'allocated':>10} {'reserved':>9} {'error':>7}   (GiB)  {'step ms':>8}")
errors = []
for st in plan["stages"]:
    s, predicted = st["stage_id"], st["memory_bytes"] / GiB
    if s not in measured:
        print(f"{s:>5} {predicted:10.2f}   (no measurement)")
        continue
    ms, allocated, reserved, _ = measured[s]
    errors.append(predicted - float(allocated))
    print(f"{s:>5} {predicted:10.2f} {float(allocated):10.2f} {float(reserved):9.2f} {errors[-1]:+7.2f}         {ms:>8}")
if errors:
    print(f"mean |error| {sum(map(abs, errors)) / len(errors):.2f} GiB, max |error| {max(map(abs, errors)):.2f} GiB "
          f"(predicted - allocated)")
