/* json.c -- minimal recursive-descent JSON parser for auto_lut.
 *
 * Stage 2 (subtask 2): parser scaffolding + primitives.
 *   - Parser struct
 *   - json_skip_ws, json_parse_string, json_parse_number
 *   - Stub public API (filled in later subtasks)
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
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            s->p++;
        } else {
            break;
        }
    }
}

static JsonValue *json_parse_value(Parser *s);  /* forward decl */

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

    free(out);
    s->error = 1;
    return NULL;
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

/* stubs -- filled in by subsequent subtasks */
static JsonValue *json_parse_value(Parser *s) { (void)s; return NULL; }
JsonValue *json_parse(const char *str) { (void)str; return NULL; }
JsonValue *json_obj_get(JsonValue *obj, const char *key) { (void)obj; (void)key; return NULL; }
double json_get_num(JsonValue *obj, const char *key, double def) { (void)obj; (void)key; return def; }
const char *json_get_str(JsonValue *obj, const char *key, const char *def) { (void)obj; (void)key; return def; }
void json_free(JsonValue *v) { (void)v; }
