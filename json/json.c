/* json.c -- minimal recursive-descent JSON parser for auto_lut.
 *
 * Stage 5 (subtask 5): accessors.
 *   - json_obj_get: key lookup in object
 *   - json_get_num: numeric accessor with default (accepts NUM and BOOL)
 *   - json_get_str: string accessor with default
 */

#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
    const char *end;
    int         error;
} Parser;

static JsonValue *new_value(int type)
{
    JsonValue *v = (JsonValue *)calloc(1, sizeof(*v));
    if (v) v->type = type;
    return v;
}

static void json_skip_ws(Parser *s)
{
    while (s->p < s->end) {
        char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { s->p++; }
        else break;
    }
}

static JsonValue *json_parse_value(Parser *s);

static char *json_parse_string(Parser *s, int *out_len)
{
    if (s->p >= s->end || *s->p != '"') { s->error = 1; return NULL; }
    s->p++;
    size_t cap = 16, len = 0;
    char  *out = (char *)malloc(cap);
    if (!out) { s->error = 1; return NULL; }
    while (s->p < s->end) {
        unsigned char c = (unsigned char)*s->p++;
        if (c == '"') {
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
                    if (len + 4 > cap) {
                        while (len + 4 > cap) cap *= 2;
                        char *t = (char *)realloc(out, cap);
                        if (!t) { free(out); s->error = 1; return NULL; }
                        out = t;
                    }
                    if (cp < 0x80) out[len++] = (char)cp;
                    else if (cp < 0x800) {
                        out[len++] = (char)(0xC0 | (cp >> 6));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        out[len++] = (char)(0xE0 | (cp >> 12));
                        out[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    continue;
                }
                default: free(out); s->error = 1; return NULL;
            }
        } else if (c < 0x20) {
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
    free(out); s->error = 1; return NULL;
}

static JsonValue *json_parse_number(Parser *s)
{
    const char *start = s->p;
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

static JsonValue *json_parse_value(Parser *s)
{
    json_skip_ws(s);
    if (s->p >= s->end) { s->error = 1; return NULL; }
    char c = *s->p;
    switch (c) {
        case '{': {
            s->p++;
            JsonValue *v = new_value(JSON_OBJ);
            if (!v) { s->error = 1; return NULL; }
            size_t cap = 0;
            json_skip_ws(s);
            if (s->p < s->end && *s->p == '}') { s->p++; return v; }
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
            s->p++;
            JsonValue *v = new_value(JSON_ARR);
            if (!v) { s->error = 1; return NULL; }
            size_t cap = 0;
            json_skip_ws(s);
            if (s->p < s->end && *s->p == ']') { s->p++; return v; }
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
        case 't':
            if (s->p + 4 <= s->end && memcmp(s->p, "true", 4) == 0) {
                s->p += 4;
                JsonValue *v = new_value(JSON_BOOL);
                if (!v) { s->error = 1; return NULL; }
                v->num = 1.0;
                return v;
            }
            s->error = 1; return NULL;
        case 'f':
            if (s->p + 5 <= s->end && memcmp(s->p, "false", 5) == 0) {
                s->p += 5;
                JsonValue *v = new_value(JSON_BOOL);
                if (!v) { s->error = 1; return NULL; }
                v->num = 0.0;
                return v;
            }
            s->error = 1; return NULL;
        case 'n':
            if (s->p + 4 <= s->end && memcmp(s->p, "null", 4) == 0) {
                s->p += 4;
                JsonValue *v = new_value(JSON_NULL);
                if (!v) { s->error = 1; return NULL; }
                return v;
            }
            s->error = 1; return NULL;
        default:
            if (c == '-' || (c >= '0' && c <= '9')) return json_parse_number(s);
            s->error = 1; return NULL;
    }
}

JsonValue *json_parse(const char *str)
{
    if (!str) return NULL;
    size_t len = strlen(str);
    if (len == 0) return NULL;
    Parser s = { str, str + len, 0 };
    JsonValue *v = json_parse_value(&s);
    if (!v) return NULL;
    json_skip_ws(&s);
    if (s.error || s.p != s.end) { json_free(v); return NULL; }
    return v;
}

/* ---- subtask 5: accessors ---- */
JsonValue *json_obj_get(JsonValue *obj, const char *key)
{
    if (!obj || obj->type != JSON_OBJ || !key) return NULL;
    for (int i = 0; i < obj->obj_len; i++) {
        if (strcmp(obj->keys[i], key) == 0)
            return obj->vals[i];
    }
    return NULL;
}

double json_get_num(JsonValue *obj, const char *key, double def)
{
    JsonValue *v = json_obj_get(obj, key);
    if (!v) return def;
    if (v->type == JSON_NUM)  return v->num;
    if (v->type == JSON_BOOL) return v->num;
    return def;
}

const char *json_get_str(JsonValue *obj, const char *key, const char *def)
{
    JsonValue *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_STR) return def;
    return v->str;
}

/* stub -- filled in by next subtask */
void json_free(JsonValue *v) { (void)v; }
