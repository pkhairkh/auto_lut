---
Task ID: tokenizer-w0-w5
Agent: Super Z (subagent for module/tokenizer)
Task: Implement BPE Tokenizer (tokenizer.h / tokenizer.c) for auto_lut project.

Work Log:
- Wave 0 (Environment provisioning):
  - SSH'd into root@34.39.106.202 via paramiko (sshpass unavailable locally).
  - Verified machine: 8 cores, 30 GiB RAM, 82 GiB free disk, gcc 14.3.1.
  - Discovered repo uses git worktrees: /root/auto_lut_<module> per branch.
  - Found json.c/json.h at /root/auto_lut_json/json/ (different API than the
    placeholder header in the task brief). Adapted tokenizer.c to use
    JSON_OBJ/JSON_ARR/JSON_STR/JSON_NUM, json_obj_get(), and the parallel
    keys[]/vals[] arrays (not the JsonObjectEntry pattern from the brief).
  - Inspected /root/dolphin/hf_model_main/tokenizer.json:
      * model.type = BPE
      * model.vocab size = 50000
      * model.merges size = 49721
      * pre_tokenizer = Sequence: Split("SPL1T-TH1S-Pl3A5E", Removed) ->
        Digits(individual_digits=true) -> Split(regex, Isolated) ->
        Split("\n", Isolated) -> ByteLevel(use_regex=true, add_prefix_space=false)
      * decoder = ByteLevel (add_prefix_space=true, use_regex=true)
      * post_processor = TemplateProcessing with <s>/</s> wrapping
      * added_tokens = 23944 entries; ids 50000..73943 include
        ' <Answer/>' (with leading space) at id 73920
  - Verified with HF tokenizers Python lib (reference only, not used in C):
      * With post_processor ON:  [0, 0, 47928, 286, 7996, 1160, 299, 495,
        5262, 36, 73920, 2]
      * With post_processor OFF: [0, 47928, 286, 7996, 1160, 299, 495,
        5262, 36, 73920]  <-- matches DoD expected output
      * Conclusion: DoD expects post_processor DISABLED (verbatim encode
        of the input text, special tokens matched directly).

- Wave 1 (Header):
  - Wrote tokenizer.h with the exact API specified in the task brief:
    Tokenizer *tokenizer_load(const char *path);
    int *tokenizer_encode(Tokenizer *tok, const char *text, int *out_len);
    char *tokenizer_decode(Tokenizer *tok, const int *ids, int len);
    void tokenizer_free(Tokenizer *tok);
  - Documented design decisions in the header (no post-processor,
    caller owns returned arrays, depends only on json.h).

- Waves 2-5 (Implementation in tokenizer.c):
  Wave 2 - Vocab loading:
    - FNV-1a hash map (HashMap struct with buckets, auto-grow at 70% load).
    - load_vocab() iterates model.vocab's parallel keys[]/vals[] arrays.
    - build_reverse_vocab() walks the hashmap to populate rev_vocab[]
      indexed by id (sized by max_id+1 = 73944).
  Wave 3 - BPE merges:
    - load_merges() stores "tokA tokB" -> rank (lower rank = higher priority).
    - bpe_apply() splits the byte-level-encoded piece into UTF-8 code points
      (each code point corresponds to one original byte), then iteratively
      finds the lowest-rank adjacent pair and merges until no pair has a rank.
  Wave 4 - encode:
    - match_special_at(): linear scan over special_str[] (sorted longest-first
      so ' <Answer/>' wins over '<Answer' fragments at any position).
    - pretokenize_segment(): applies the Sequence stages 1-4 in one pass
      (SPL1T literal, individual digits, bracket/punct isolation, newlines),
      accumulating "run" pieces that are then fed to bytlevel_split().
    - bytlevel_split(): ASCII-focused GPT-2 regex (contractions, ' ?\p{L}+,
      ' ?\p{N}+, ' ?[^\s\p{L}\p{N}]+, \s+(?!\S), \s+). Each piece is
      converted via bytes_to_bytestr() (GPT-2 byte-to-unicode table) before
      being passed to bpe_apply().
  Wave 5 - decode:
    - tokenizer_decode() concatenates rev_vocab[id] for all ids, then runs
      bytestr_to_bytes() which reverses the byte-level map (code point ->
      byte) to produce the final UTF-8 string.

- Compile: gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra -c tokenizer.c
  -> 0 errors, 0 warnings.

Stage Summary:
- tokenizer.h committed (2927 bytes) with public API.
- tokenizer.c committed (34992 bytes) implementing Waves 2-5:
  vocab hash map, merge hash map, special-token matching, ByteLevel
  pre-tokenizer, BPE merges, encode, decode, free.
- Build artifact: tokenizer.o (16152 bytes).
- Next: Wave 6 (test_tokenizer.c) and Wave 7 (push, DoD verify).

---
Task ID: tasklist-07-wave6
Agent: auto_lut forward orchestrator (main)
Task: Wave 6 - Implement metadata.h/.c + main.c orchestrator + Makefile.

Work Log:
- Authored forward/metadata.h + metadata.c: writer for metadata_coreml_pg.json.
  Walks every SafeTensors tensor, classifies palettizable (2D F16/BF16/F32 weight)
  vs non-palettizable (1D biases, I64 index tables, etc.), emits JSON with
  name/dtype/ndim/shape/n_elements/byte_size/is_2d/is_palettizable plus
  captured activation statistics (n_samples, in_dim, hessian_mean, hessian_max)
  for tensors that have activations.
- Authored main.c orchestrator:
  * Loads model.safetensors, config.json, preprocessor_config.json, tokenizer.json.
  * Parses preprocessor config for image_mean/std/rescale/size.
  * Glob-expands images_glob (shell-style *) into a list of PNG paths.
  * Caps at 4 images to stay under 120s time budget.
  * Preprocesses each PNG via png_load + preprocess_donut -> CHW float32.
  * Builds decoder prompt: <s> </s> </s> </s> </s> (5 tokens, resolved via tokenizer).
  * Calls forward_run() to capture matmul input activations.
  * Calls metadata_write() to emit metadata_coreml_pg.json.
  * Palettizes every 2D F16 weight tensor: 4-bit uniform quantization based on
    tensor min/max, indices written transposed (in_dim, out_dim) per ANE
    pre-transpose convention, packed via pack_idx4 (LSB-first), LUT written
    as 16-entry FP16 .lut_scalar file.
  * Copies non-palettizable tensors (biases, norms, I64 tables) as raw FP16 .fp16 files.
- Authored Makefile: builds all modules (safetensors, fp16, png, tokenizer,
  preprocess, pack, forward) + auto_lut binary. Uses gcc -O3 -march=native
  -fopenmp -std=c11 -Wall -Wextra. Generic pattern rule for *.c -> *.o,
  module-specific dependency declarations, test_deps target, clean target.

Stage Summary:
- All Wave 6 source files written.
- Ready to build and run integration test (Wave 7).
