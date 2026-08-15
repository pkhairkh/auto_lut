---
Task ID: tasklist-07-wave1
Agent: auto_lut forward orchestrator (main)
Task: Wave 1 — Implement missing dependency modules (fp16, png, tokenizer, preprocess, pack.c) required by forward pass.

Work Log:
- SSH into root@34.39.106.202 (8 cores, 30 GB RAM, 82 GB free, gcc 14.3.1).
- Branch module/forward was empty (initial commit). Consolidated from sibling branches:
  safetensors/ from module/preprocess; png/ from module/png; pack/ from module/pack.
- Authored fp16/fp16.h + fp16.c (IEEE 754 binary16 conversion, bit-exact RNE).
- Authored tokenizer/tokenizer.h + tokenizer.c (HF added_tokens loader).
- Authored preprocess/preprocess.h + preprocess.c (Donut-style resize/rescale/normalize -> CHW f32).
- Fixed critical fp16 bug: original code shifted mantissa LEFT by 13 bits (overflowing uint32
  and corrupting rounding-bit extraction); rewrote to operate on original 23-bit mantissa directly.
- test_deps: 7/7 PASS against real Dolphin data.

Stage Summary:
- All 6 dependency modules compile clean with gcc -O3 -march=native -fopenmp -std=c11.
- Committed to module/forward.
