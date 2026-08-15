---
Task ID: tokenizer-w0-w7
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
    keys[]/vals[] arrays.
  - Inspected /root/dolphin/hf_model_main/tokenizer.json:
      * model.type = BPE
      * model.vocab size = 50000
      * model.merges size = 49721 (stored as 2-element arrays ["tokA","tokB"],
        not as "tokA tokB" strings - the task brief was wrong about this)
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
  - Wrote tokenizer.h with the exact API specified in the task brief.

- Waves 2-5 (Implementation in tokenizer.c):
  Wave 2 - Vocab loading:
    - FNV-1a hash map (HashMap struct with buckets, auto-grow at 70% load).
    - load_vocab() iterates model.vocab's parallel keys[]/vals[] arrays.
    - build_reverse_vocab() walks the hashmap to populate rev_vocab[] indexed
      by id (sized by max_id+1 = 73944).
  Wave 3 - BPE merges:
    - load_merges() handles BOTH formats: "tokA tokB" string OR
      ["tokA", "tokB"] 2-element array (the actual format in tokenizer.json).
      Builds "tokA tokB" key -> rank (lower rank = higher priority).
    - bpe_apply() splits the byte-level-encoded piece into UTF-8 code points
      (each code point corresponds to one original byte), then iteratively
      finds the lowest-rank adjacent pair and merges until no pair has a rank.
  Wave 4 - encode:
    - match_special_at(): linear scan over special_str[] (sorted longest-first
      so ' <Answer/>' wins over '<Answer' fragments at any position).
      Skips 0-length specials (whose content contains a NUL byte, e.g.
      added_tokens with content '\x00' or '\x01' at ids 67043/69369 - these
      cannot be matched from text because strlen returns 0).
    - pretokenize_segment(): applies the Sequence stages 1-4 in one pass
      (SPL1T literal, individual digits, bracket/punct isolation, newlines),
      accumulating "run" pieces that are then fed to bytlevel_split().
    - bytlevel_split(): ASCII-focused GPT-2 regex (contractions, ' ?\p{L}+,
      ' ?\p{N}+, ' ?[^\s\p{L}\p{N}]+, \s+(?!\S), \s+). Each piece is
      converted via bytes_to_bytestr() (GPT-2 byte-to-unicode table) before
      being passed to bpe_apply().
    - Safety: encode loop advances i by max(sp_len, 1) after a special match
      to guarantee termination even if a 0-length special is somehow matched.
  Wave 5 - decode:
    - tokenizer_decode() looks up rev_vocab[id] for each id.
    - Special tokens are emitted verbatim (no byte-level decode) since
      their content is already in final form (e.g. " <Answer/>").
    - Regular tokens are reversed via bytestr_to_bytes() (code point -> byte).
    - is_special_lookup() helper does linear scan over special_id[].

- Wave 6 (Test - test_tokenizer.c):
  - Loads /root/dolphin/hf_model_main/tokenizer.json
  - Encodes "<s>Parse the reading order of this document. <Answer/>"
  - Asserts output == [0, 47928, 286, 7996, 1160, 299, 495, 5262, 36, 73920]
  - Decodes the IDs back to text and asserts equality with the original prompt.
  - Result:
      ENCODE: PASS (IDs match expected exactly)
      DECODE: PASS (round-trip matches prompt exactly)
      ALL CHECKS PASS

- Wave 7 (Commit & push):
  - Created dedicated worktree /root/auto_lut_tok for module/tokenizer branch
    (parallel sub-agents were switching branches out from under us in the
    shared /root/auto_lut working tree).
  - Committed tokenizer.h, tokenizer.c, test_tokenizer.c, json.c, json.h
    on module/tokenizer branch.
  - Pushed to origin/module/tokenizer.

Stage Summary:
- DoD MET: Tokenizer produces correct IDs for the Dolphin prompt.
- DoD MET: Committed to module/tokenizer branch, pushed to origin.
- Build: gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra (0 warnings).
- Performance: load 0.85s, encode 0.005s for the test prompt.
- Round-trip is byte-perfect: encode then decode reproduces the input.
- Files: tokenizer.h (2927 B), tokenizer.c (37899 B), test_tokenizer.c (3689 B).
