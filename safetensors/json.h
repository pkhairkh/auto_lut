#ifndef JSON_H
#define JSON_H

#include <stddef.h>

/* Minimal recursive-descent JSON parser used by the auto_lut safetensors
 * loader. Sufficient for parsing safetensors headers and small model
 * config files; not a fully spec-conformant production parser.
 *
 * Memory model: every JsonValue owns its children. Call json_free() on
 * the root to release the entire tree.
 */

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

struct _JsonObjectEntry;

typedef struct JsonValue {
    JsonType type;
    union {
        int           boolean;   /* JSON_BOOL   */
        double        number;    /* JSON_NUMBER */
        char         *string;    /* JSON_STRING (NUL-terminated, owned) */
        struct {
            struct JsonValue **items;
            size_t count;
        } array;                /* JSON_ARRAY  */
        struct {
            struct _JsonObjectEntry *entries;
            size_t count;
        } object;               /* JSON_OBJECT */
    } v;
} JsonValue;

typedef struct _JsonObjectEntry {
    char      *key;             /* owned */
    JsonValue *value;           /* owned */
} JsonObjectEntry;

/* Parse a NUL-terminated JSON document. Returns NULL on parse error. */
JsonValue *json_parse(const char *text);

/* Parse a JSON document of length `len` (may contain embedded NULs in
 * string escapes, but typically text[len-1] should be a valid char).
 * Returns NULL on parse error. */
JsonValue *json_parse_n(const char *text, size_t len);

/* Recursively free a JsonValue tree. Safe on NULL. */
void json_free(JsonValue *v);

/* --- accessors --- */

/* Look up an object member by key. Returns NULL if not found or if v
 * is not an object. */
const JsonValue *json_object_get(const JsonValue *v, const char *key);

/* Array element by index. Returns NULL if out of bounds or not array. */
const JsonValue *json_array_get(const JsonValue *v, size_t index);

/* Convenience: parse a JSON_NUMBER as long. Returns 0 on non-number. */
long json_as_long(const JsonValue *v);

/* Convenience: copy a JSON_STRING into a caller-supplied buffer of size
 * `cap` bytes (NUL-terminated). Returns the string length, or -1 if v
 * is not a string. Truncates with NUL if too long. */
int json_as_string(const JsonValue *v, char *out, size_t cap);

#endif /* JSON_H */
