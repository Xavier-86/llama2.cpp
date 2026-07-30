#!/usr/bin/env python3
"""Insert `(void)0;` into multi-line function bodies that contain no
statements (only comments/whitespace), so a breakpoint can be set there.

A function-body open brace is identified heuristically: a `{` whose previous
non-space *code* character is `)`. Comment and string regions are masked so
braces inside them are ignored. Works on raw bytes so multi-byte UTF-8 in
comments cannot shift offsets.
"""
import sys

SLASH, STAR = 47, 42
DQUOTE, SQUOTE, BACKSLASH = 34, 39, 92
LBRACE, RBRACE, RPAREN = 123, 125, 41
NL, CR, TAB, SPACE = 10, 13, 9, 32


def mask_code(b):
    """Return a bytearray where comment/string/char bytes are spaces."""
    m = bytearray(b)
    i, n = 0, len(b)
    state = "code"  # code | line | block | str | chr
    while i < n:
        c = b[i]
        if state == "code":
            if c == SLASH and i + 1 < n and b[i + 1] == SLASH:
                m[i] = m[i + 1] = SPACE
                i += 2
                state = "line"
                continue
            if c == SLASH and i + 1 < n and b[i + 1] == STAR:
                m[i] = m[i + 1] = SPACE
                i += 2
                state = "block"
                continue
            if c == DQUOTE:
                m[i] = SPACE
                i += 1
                state = "str"
                continue
            if c == SQUOTE:
                m[i] = SPACE
                i += 1
                state = "chr"
                continue
            i += 1
        elif state == "line":
            if c == NL:
                state = "code"
            else:
                m[i] = SPACE
            i += 1
        elif state == "block":
            if c == STAR and i + 1 < n and b[i + 1] == SLASH:
                m[i] = m[i + 1] = SPACE
                i += 2
                state = "code"
            else:
                if c != NL:
                    m[i] = SPACE
                i += 1
        else:  # str | chr
            quote = DQUOTE if state == "str" else SQUOTE
            if c == BACKSLASH:
                m[i] = SPACE
                if i + 1 < n:
                    m[i + 1] = SPACE
                i += 2
                continue
            if c == quote:
                m[i] = SPACE
                i += 1
                state = "code"
                continue
            if c != NL:
                m[i] = SPACE
            i += 1
    return m


def process(b):
    mask = mask_code(b)
    n = len(b)
    inserts = []  # (pos_of_closing_brace, indent_bytes)
    i = 0
    while i < n:
        if mask[i] == LBRACE:
            j = i - 1
            while j >= 0 and mask[j] in (SPACE, TAB, NL, CR):
                j -= 1
            if j < 0 or mask[j] != RPAREN:
                i += 1
                continue
            depth = 1
            k = i + 1
            while k < n and depth:
                if mask[k] == LBRACE:
                    depth += 1
                elif mask[k] == RBRACE:
                    depth -= 1
                k += 1
            if depth:
                i += 1
                continue
            close = k - 1
            body = mask[i + 1:close]
            if body.strip() == b"" and NL in body:
                ls = b.rfind(b"\n", 0, close) + 1
                base = b[ls:close]
                base = base[: len(base) - len(base.lstrip())]
                inserts.append((close, base))
            i = k
        else:
            i += 1
    for close, base in reversed(inserts):
        # b[:close] already ends with the closing brace's indentation,
        # so the statement goes one level deeper and the brace keeps `base`.
        b = b[:close] + b"    (void)0;\n" + base + b[close:]
    return b, len(inserts)


def main():
    total = 0
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            data = f.read()
        new, count = process(data)
        if count:
            with open(path, "wb") as f:
                f.write(new)
        total += count
        print(f"{path}: {count} function(s) filled")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
