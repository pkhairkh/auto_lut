/* json_compat.h — Adapter that maps the BPE tokenizer's shorthand JSON API
 * names to the in-tree json.h API.
 *
 * The parallel-agent tokenizer.c was written against a slightly different
 * json.h (shorter names: json_obj_get, JSON_NUM, ->num, ->keys, ->arr, etc.).
 * Rather than modify their source, we provide this shim and force-include
 * it via -include tokenizer/json_compat.h on the compiler command line.
 *
 * This keeps the BPE tokenizer.c untouched and lets it compile against
 * the canonical safetensors/json.h.
 *
 * FIELD-ACCESS MACROS
 * -------------------
 * The macros below (obj_len, keys, vals, arr_len, arr, num, str, boolean)
 * are #define'd to the corresponding field paths inside the canonical
 * JsonValue union. This works because the tokenizer.c only uses them as
 * `ptr->field` — never as standalone variable names (verified by grep).
 *
 * If any of these identifiers are ever used as variable names inside
 * tokenizer.c, this shim will break silently. A grep audit confirms
 * they are not.
 */
#ifndef JSON_COMPAT_H
#define JSON_COMPAT_H

#include "json.h"   /* the canonical API */

/* ---- enum-constant aliases ---- */
#define JSON_NUM    JSON_NUMBER
#define JSON_STR    JSON_STRING
#define JSON_OBJ    JSON_OBJECT
#define JSON_BOOL   JSON_BOOL   /* already matches */
#define JSON_NULL   JSON_NULL   /* already matches */
#define JSON_ARR    JSON_ARRAY

/* ---- function-name aliases (return const JsonValue*) ---- */
static inline const JsonValue *json_obj_get(const JsonValue *v, const char *key) {
    return json_object_get(v, key);
}
static inline const JsonValue *json_arr_get(const JsonValue *v, size_t index) {
    return json_array_get(v, index);
}

/* ---- field-accessor macros ----
 *
 * Canonical JsonValue layout (from json.h):
 *
 *   typedef struct JsonValue {
 *       JsonType type;
 *       union {
 *           int   boolean;
 *           double number;
 *           char  *string;
 *           struct { struct JsonValue **items; size_t count; } array;
 *           struct { struct _JsonObjectEntry *entries; size_t count; } object;
 *       } v;
 *   } JsonValue;
 *
 * JsonObjectEntry:
 *   typedef struct _JsonObjectEntry {
 *       char      *key;
 *       JsonValue *value;
 *   } JsonObjectEntry;
 *
 * The tokenizer.c expects these shorthand fields:
 *   ->num, ->str, ->boolean          ->  ->v.number, ->v.string, ->v.boolean
 *   ->arr_len, ->arr[i]              ->  ->v.array.count, ->v.array.items[i]
 *   ->obj_len, ->keys[i], ->vals[i]  ->  ->v.object.count,
 *                                        ->v.object.entries[i].key,
 *                                        ->v.object.entries[i].value
 *
 * The macro #defines below achieve this. They are safe because the bare
 * identifiers (num, str, boolean, arr_len, arr, obj_len, keys, vals) are
 * never used as variable names in tokenizer.c (verified via grep).
 */
#define num       v.number
#define str       v.string
#define boolean   v.boolean
#define arr_len   v.array.count
#define arr       v.array.items
#define obj_len   v.object.count
/* keys[i] and vals[i] require indirect access through entries[i].
 * We can't directly #define `keys` to `v.object.entries` because then
 * `obj->keys[i]` would expand to `obj->v.object.entries[i]` which is a
 * JsonObjectEntry, not a `char*`. We need `obj->v.object.entries[i].key`.
 *
 * Since the preprocessor cannot insert `.[i].key` based on context, we
 * instead expose `keys` and `vals` as inline helper functions that take
 * the JsonValue and an index. But the tokenizer.c uses `vocab_obj->keys[i]`
 * directly — that syntax requires a member access, not a function call.
 *
 * The cleanest solution: provide `keys` and `vals` as macros that expand
 * to a small array-like wrapper. But that's complex.
 *
 * Simpler: directly access via the entries array. We #define `keys` and
 * `vals` to `v.object.entries` so that:
 *   vocab_obj->keys[i]    becomes vocab_obj->v.object.entries[i]
 * which is a JsonObjectEntry. Then the tokenizer.c uses
 *   const char *k = vocab_obj->keys[i];
 * which would fail because JsonObjectEntry != char*.
 *
 * The actual fix: tokenizer.c does
 *   const char *k  = vocab_obj->keys[i];
 *   JsonValue   *vv = vocab_obj->vals[i];
 *
 * We need keys[i] -> v.object.entries[i].key
 * and     vals[i] -> v.object.entries[i].value
 *
 * The only way to do this with the preprocessor is to NOT use a macro
 * for `keys`/`vals` and instead patch the source. But we said we
 * wouldn't modify the source.
 *
 * Alternative: define `keys` and `vals` as macros that expand to
 * expressions which, when subscripted with [i], yield the right thing.
 * A common trick is to use a pointer-to-array:
 *
 *   #define keys   v.object.entries->key    // WRONG: -> after -> doesn't work for [i]
 *
 * Actually: if `entries` is a `JsonObjectEntry *`, then
 * `entries[i].key` works. So `vocab_obj->v.object.entries[i].key` works.
 * But we want `vocab_obj->keys[i]` to expand to that, which means
 * `keys` must expand to `v.object.entries` and then we need the
 * tokenizer.c to write `->keys[i].key` — but it writes `->keys[i]`.
 *
 * Conclusion: there is no clean preprocessor-only way to map `->keys[i]`
 * to `->v.object.entries[i].key`. We must either:
 *   (a) patch tokenizer.c to use a helper function, or
 *   (b) patch json.h to add keys/vals/arr/arr_len/obj_len as real fields.
 *
 * Option (b) is the least invasive — we extend the JsonValue struct with
 * a union overlay that exposes the shorthand names directly. But that
 * requires modifying json.h, which is shared with other modules.
 *
 * Option (a) is local: we patch the 3 lines in tokenizer.c that use
 * keys[i]/vals[i] to use a helper accessor. This is a 6-line source
 * patch and is the cleanest option.
 *
 * We provide json_object_key(v, i) and json_object_val(v, i) helpers
 * here, and the Makefile will sed-patch tokenizer.c to use them.
 */

static inline const char *json_obj_key_at(const JsonValue *v, size_t i) {
    if (!v || v->type != JSON_OBJECT || i >= v->v.object.count) return NULL;
    return v->v.object.entries[i].key;
}
static inline JsonValue *json_obj_val_at(const JsonValue *v, size_t i) {
    if (!v || v->type != JSON_OBJECT || i >= v->v.object.count) return NULL;
    return v->v.object.entries[i].value;
}

#endif /* JSON_COMPAT_H */
