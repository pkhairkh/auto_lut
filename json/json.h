/* json.h -- minimal recursive-descent JSON parser for auto_lut.
 *
 * Parses safetensors headers (<1MB) and HuggingFace config.json files.
 * Pure C11, no external dependencies. Entire input must fit in memory.
 *
 * Memory model: json_parse() returns a malloc'd tree. Caller frees
 * the whole tree with json_free(root). Use json_obj_get() to traverse,
 * json_get_num()/json_get_str() for typed access with defaults.
 */
#ifndef JSON_H
#define JSON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JsonValue JsonValue;
struct JsonValue {
    enum { JSON_NULL, JSON_BOOL, JSON_NUM, JSON_STR, JSON_ARR, JSON_OBJ } type;
    char *str;        /* NUL-terminated; valid when type == JSON_STR       */
    int   str_len;   /* length without NUL; valid when type == JSON_STR    */
    double num;      /* numeric value; for JSON_BOOL stores 1.0/0.0       */
    JsonValue **arr; /* array items; valid when type == JSON_ARR          */
    int   arr_len;
    char **keys;     /* parallel array of object keys (NUL-terminated)     */
    JsonValue **vals;/* parallel array of object values                   */
    int   obj_len;   /* number of object entries; valid when type==JSON_OBJ */
};

/* Parse a NUL-terminated JSON document. Returns NULL on syntax error.
 * Top-level value may be object, array, string, number, bool, or null.
 * The returned tree must be freed with json_free(). */
JsonValue *json_parse(const char *str);

/* Lookup a key in a JSON object. Returns NULL if obj is not an object
 * or the key is absent. The returned pointer is owned by the tree. */
JsonValue *json_obj_get(JsonValue *obj, const char *key);

/* Convenience accessor: returns the numeric value at `key` of an object.
 * Accepts JSON_NUM and JSON_BOOL (true -> 1.0, false -> 0.0).
 * Returns `def` if obj is not an object, the key is missing, or the
 * value is not numeric. */
double json_get_num(JsonValue *obj, const char *key, double def);

/* Convenience accessor: returns the string value at `key` of an object.
 * Returns `def` if obj is not an object, the key is missing, or the
 * value is not a string. The returned pointer is owned by the tree;
 * the caller must NOT free it. */
const char *json_get_str(JsonValue *obj, const char *key, const char *def);

/* Recursively free a JsonValue tree. Safe to call on NULL. */
void json_free(JsonValue *v);

#ifdef __cplusplus
}
#endif

#endif /* JSON_H */
