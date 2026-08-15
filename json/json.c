/* json.c -- minimal recursive-descent JSON parser for auto_lut.
 *
 * Design goals:
 *   - Single C11 translation unit, no external dependencies.
 *   - Strict enough for safetensors headers and HF config.json files.
 *   - Tolerant of leading/trailing whitespace; rejects malformed input.
 *
 * Conformance:
 *   - Strings: "..." with \\ escape handling (\n \t \" \\ \/ \b \f \r
 *     and \uXXXX -> UTF-8 BMP encoding).
 *   - Numbers: integers, decimals, scientific notation, parsed via strtod().
 *   - Objects: { "key": value, ... } with duplicate keys allowed (last wins).
 *   - Arrays:  [ value, ... ]
 *   - Literals: true / false / null
 *
 * Memory: all allocations use malloc()/realloc()/calloc(). The caller
 * frees the whole tree with json_free(root).
 */

#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/* parser state                                                       */
/* ================================================================== */

typedef struct {
    const char *p;     /* current cursor */
    const char *end;    /* one past last byte */
    int         error;  /* set to 1 on parse failure */
} Parser;

/* ------------------------------------------------------------------ */
/* json_skip_ws: skip space, tab, newline, CR.                        */
/* Spec rule: per RFC 8259, ws = %x20 / %x09 / %x0A / %x0D.           */
/* ------------------------------------------------------------------ */
static void json_skip_ws(Parser *s)
{
    while (s->p < s->end) {
        char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            s->p++;
        } else {
            break;
        }
    }
}

/* Forward declaration: parse_value is the recursive entry point. */
static JsonValue *json_parse_value(Parser *s);

/* ================================================================== */
/* small helpers                                                      */
/* ================================================================== */

/* Allocate a new JsonValue of the given type. Fields are zero-initialized. */
static JsonValue *new_value(int type)
{
    JsonValue *v = (JsonValue *)calloc(1, sizeof(*v));
    if (v) v->type = type;
    return v;
}

/* ------------------------------------------------------------------ */
/* json_parse_string: parse a "..." JSON string literal.               */
/* Handles \\ escapes: \" \\ \/ \b \f \n \r \t and \uXXXX (BMP only).  */
/* Returns a freshly malloc'd NUL-terminated string in *out_str and    */
/* its byte length (excl. NUL) in *out_len, or NULL on error.          */
/* Advances s->p past the closing quote.                              */
/* ------------------------------------------------------------------ */
static char *json_parse_string(Parser *s, int *out_len)
{
    if (s->p >= s->end || *s->p != '"') { s->error = 1; return NULL; }
    s->p++; /* skip opening quote */

    size_t cap = 16, len = 0;
    char  *out = (char *)malloc(cap);
    if (!out) { s->error = 1; return NULL; }

    while (s->p < s->end) {
        unsigned char c = (unsigned char)*s->p++;

        if (c == '"') {
            /* ensure room for NUL */
            if (len + 1 > cap) {
                char *t = (char *)realloc(out, len + 1);
                if (!t) { free(out); s->error = 1; return NULL; }
                out = t;
            }
            out[len] = '\0';
            if (out_len) *out_len = (int)len;
            return out;
        }

        if (c == '\\') {
            if (s->p >= s->end) { free(out); s->error = 1; return NULL; }
            char esc = *s->p++;
            switch (esc) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u': {
                    /* \uXXXX -> UTF-8 (BMP only, no surrogate-pair merging) */
                    if (s->p + 4 > s->end) { free(out); s->error = 1; return NULL; }
                    unsigned int cp = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = s->p[i];
                        unsigned int d;
                        if (h >= '0' && h <= '9')      d = (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                        else { free(out); s->error = 1; return NULL; }
                        cp = (cp << 4) | d;
                    }
                    s->p += 4;
                    /* ensure capacity for up to 3 UTF-8 bytes + NUL */
                    if (len + 4 > cap) {
                        while (len + 4 > cap) cap *= 2;
                        char *t = (char *)realloc(out, cap);
                        if (!t) { free(out); s->error = 1; return NULL; }
                        out = t;
                    }
                    if (cp < 0x80) {
                        out[len++] = (char)cp;
                    } else if (cp < 0x800) {
                        out[len++] = (char)(0xC0 | (cp >> 6));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        out[len++] = (char)(0xE0 | (cp >> 12));
                        out[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    continue;
                }
                default:
                    free(out); s->error = 1; return NULL;
            }
        } else if (c < 0x20) {
            /* unescaped control character not allowed */
            free(out); s->error = 1; return NULL;
        }

        if (len + 2 > cap) {
            cap *= 2;
            char *t = (char *)realloc(out, cap);
            if (!t) { free(out); s->error = 1; return NULL; }
            out = t;
        }
        out[len++] = (char)c;
    }

    /* ran off end before closing quote */
    free(out);
    s->error = 1;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* json_parse_number: parse a JSON numeric literal using strtod().    */
/* Validates the leading character is '-' or [0-9]; strtod() handles   */
/* the rest. Stores the value in a new JSON_NUM JsonValue.            */
/* ------------------------------------------------------------------ */
static JsonValue *json_parse_number(Parser *s)
{
    const char *start = s->p;

    /* Permissive scan: gather all chars that could be part of a number
     * (digit, sign, decimal point, exponent markers), then feed to strtod.
     * strtod will fail if the buffer isn't a valid number. */
    if (s->p < s->end && *s->p == '-') s->p++;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') s->p++;
    if (s->p < s->end && *s->p == '.') {
        s->p++;
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') s->p++;
    }
    if (s->p < s->end && (*s->p == 'e' || *s->p == 'E')) {
        s->p++;
        if (s->p < s->end && (*s->p == '+' || *s->p == '-')) s->p++;
        while (s->p < s->end && *s->p >= '0' && *s->p <= '9') s->p++;
    }

    if (s->p == start) { s->error = 1; return NULL; }

    /* strtod needs a NUL-terminated buffer */
    char buf[64];
    size_t n = (size_t)(s->p - start);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, start, n);
    buf[n] = '\0';

    char *endp = NULL;
    double d = strtod(buf, &endp);
    if (endp == buf || *endp != '\0') { s->error = 1; return NULL; }

    JsonValue *v = new_value(JSON_NUM);
    if (!v) { s->error = 1; return NULL; }
    v->num = d;
    return v;
}

/* ================================================================== */
/* json_parse_value: dispatch by first character.                      */
/* ================================================================== */
static JsonValue *json_parse_value(Parser *s)
{
    json_skip_ws(s);
    if (s->p >= s->end) { s->error = 1; return NULL; }
    char c = *s->p;

    switch (c) {
        case '{': {
            /* Object: { "key": value, ... } */
            s->p++; /* skip '{' */
            JsonValue *v = new_value(JSON_OBJ);
            if (!v) { s->error = 1; return NULL; }

            size_t cap = 0;
            json_skip_ws(s);
            if (s->p < s->end && *s->p == '}') { s->p++; return v; } /* empty */

            for (;;) {
                json_skip_ws(s);
                if (s->p >= s->end || *s->p != '"') { json_free(v); s->error = 1; return NULL; }

                int key_len = 0;
                char *key = json_parse_string(s, &key_len);
                if (!key) { json_free(v); return NULL; }

                json_skip_ws(s);
                if (s->p >= s->end || *s->p != ':') { free(key); json_free(v); s->error = 1; return NULL; }
                s->p++;
                json_skip_ws(s);

                JsonValue *val = json_parse_value(s);
                if (!val) { free(key); json_free(v); return NULL; }

                if (v->obj_len == (int)cap) {
                    size_t newcap = cap ? cap * 2 : 8;
                    char **nk = (char **)realloc(v->keys, newcap * sizeof(*nk));
                    if (nk) v->keys = nk;
                    JsonValue **nv = (JsonValue **)realloc(v->vals, newcap * sizeof(*nv));
                    if (nv) v->vals = nv;
                    if (!nk || !nv) { free(key); json_free(val); json_free(v); s->error = 1; return NULL; }
                    cap = newcap;
                }
                v->keys[v->obj_len] = key;
                v->vals[v->obj_len] = val;
                v->obj_len++;

                json_skip_ws(s);
                if (s->p >= s->end) { json_free(v); s->error = 1; return NULL; }
                if (*s->p == ',') { s->p++; continue; }
                if (*s->p == '}') { s->p++; return v; }
                json_free(v); s->error = 1; return NULL;
            }
        }

        case '[': {
            /* Array: [ value, ... ] */
            s->p++; /* skip '[' */
            JsonValue *v = new_value(JSON_ARR);
            if (!v) { s->error = 1; return NULL; }

            size_t cap = 0;
            json_skip_ws(s);
            if (s->p < s->end && *s->p == ']') { s->p++; return v; } /* empty */

            for (;;) {
                JsonValue *item = json_parse_value(s);
                if (!item) { json_free(v); return NULL; }

                if (v->arr_len == (int)cap) {
                    size_t newcap = cap ? cap * 2 : 8;
                    JsonValue **t = (JsonValue **)realloc(v->arr, newcap * sizeof(*t));
                    if (!t) { json_free(item); json_free(v); s->error = 1; return NULL; }
                    v->arr = t;
                    cap = newcap;
                }
                v->arr[v->arr_len++] = item;

                json_skip_ws(s);
                if (s->p >= s->end) { json_free(v); s->error = 1; return NULL; }
                if (*s->p == ',') { s->p++; continue; }
                if (*s->p == ']') { s->p++; return v; }
                json_free(v); s->error = 1; return NULL;
            }
        }

        case '"': {
            JsonValue *v = new_value(JSON_STR);
            if (!v) { s->error = 1; return NULL; }
            int slen = 0;
            v->str = json_parse_string(s, &slen);
            if (!v->str) { free(v); return NULL; }
            v->str_len = slen;
            return v;
        }

        case 't': /* true */
            if (s->p + 4 <= s->end && memcmp(s->p, "true", 4) == 0) {
                s->p += 4;
                JsonValue *v = new_value(JSON_BOOL);
                if (!v) { s->error = 1; return NULL; }
                v->num = 1.0;
                return v;
            }
            s->error = 1; return NULL;

        case 'f': /* false */
            if (s->p + 5 <= s->end && memcmp(s->p, "false", 5) == 0) {
                s->p += 5;
                JsonValue *v = new_value(JSON_BOOL);
                if (!v) { s->error = 1; return NULL; }
                v->num = 0.0;
                return v;
            }
            s->error = 1; return NULL;

        case 'n': /* null */
            if (s->p + 4 <= s->end && memcmp(s->p, "null", 4) == 0) {
                s->p += 4;
                JsonValue *v = new_value(JSON_NULL);
                if (!v) { s->error = 1; return NULL; }
                return v;
            }
            s->error = 1; return NULL;

        default:
            /* number: '-' or [0-9] */
            if (c == '-' || (c >= '0' && c <= '9')) return json_parse_number(s);
            s->error = 1; return NULL;
    }
}

/* ================================================================== */
/* json_parse: public entry point.                                     */
/* Accepts a NUL-terminated JSON document. Top-level value may be      */
/* any of: object, array, string, number, bool, null.                  */
/* Returns NULL on syntax error or empty input.                       */
/* ================================================================== */
JsonValue *json_parse(const char *str)
{
    if (!str) return NULL;
    size_t len = strlen(str);
    if (len == 0) return NULL;

    Parser s = { str, str + len, 0 };
    JsonValue *v = json_parse_value(&s);
    if (!v) return NULL;

    /* allow trailing whitespace only */
    json_skip_ws(&s);
    if (s.error || s.p != s.end) {
        json_free(v);
        return NULL;
    }
    return v;
}

/* ================================================================== */
/* accessors                                                          */
/* ================================================================== */

/* Lookup a key in an object. Returns NULL if not found or not an object. */
JsonValue *json_obj_get(JsonValue *obj, const char *key)
{
    if (!obj || obj->type != JSON_OBJ || !key) return NULL;
    for (int i = 0; i < obj->obj_len; i++) {
        if (strcmp(obj->keys[i], key) == 0)
            return obj->vals[i];
    }
    return NULL;
}

/* Numeric accessor with default. Accepts JSON_NUM and JSON_BOOL. */
double json_get_num(JsonValue *obj, const char *key, double def)
{
    JsonValue *v = json_obj_get(obj, key);
    if (!v) return def;
    if (v->type == JSON_NUM)  return v->num;
    if (v->type == JSON_BOOL) return v->num; /* true=1.0, false=0.0 */
    return def;
}

/* String accessor with default. Returns internal pointer (not owned). */
const char *json_get_str(JsonValue *obj, const char *key, const char *def)
{
    JsonValue *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_STR) return def;
    return v->str;
}

/* ================================================================== */
/* json_free: recursively free a JsonValue tree.                       */
/* ================================================================== */
void json_free(JsonValue *v)
{
    if (!v) return;

    switch (v->type) {
        case JSON_STR:
            free(v->str);
            break;

        case JSON_ARR:
            for (int i = 0; i < v->arr_len; i++)
                json_free(v->arr[i]);
            free(v->arr);
            break;

        case JSON_OBJ:
            for (int i = 0; i < v->obj_len; i++) {
                free(v->keys[i]);
                json_free(v->vals[i]);
            }
            free(v->keys);
            free(v->vals);
            break;

        case JSON_NULL:
        case JSON_BOOL:
        case JSON_NUM:
        default:
            /* no extra allocations */
            break;
    }

    free(v);
}
