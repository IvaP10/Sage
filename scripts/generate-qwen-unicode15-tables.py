#!/usr/bin/env python3
"""Generate the Unicode 15 tables owned by sage-qwen-tokenizer.

Only Python's standard-library Unicode database is used at development time.
The tokenizer runtime has no Python or Unicode-library dependency.
"""

from __future__ import annotations

import argparse
import pathlib
import unicodedata


EXPECTED_UNICODE_VERSION = "15.0.0"
MAX_CODEPOINT = 0x10FFFF


def ranges(predicate):
    result = []
    start = None
    last = None
    for codepoint in range(MAX_CODEPOINT + 1):
        if predicate(chr(codepoint)):
            if start is None:
                start = codepoint
            last = codepoint
        elif start is not None:
            result.append((start, last))
            start = None
            last = None
    if start is not None:
        result.append((start, last))
    return result


def render_pairs(name, rows, tuple_width):
    lines = [f"pub const {name}: &[{tuple_width}] = &["]
    for row in rows:
        lines.append("    (" + ", ".join(f"0x{value:X}" for value in row) + "),")
    lines.append("];\n")
    return "\n".join(lines)


def generate():
    if unicodedata.unidata_version != EXPECTED_UNICODE_VERSION:
        raise SystemExit(
            f"Python Unicode database is {unicodedata.unidata_version}; "
            f"expected {EXPECTED_UNICODE_VERSION}"
        )

    letter_mark_number = ranges(lambda char: unicodedata.category(char)[0] in "LMN")
    letters = ranges(lambda char: unicodedata.category(char)[0] == "L")
    marks = ranges(lambda char: unicodedata.category(char)[0] == "M")
    numbers = ranges(lambda char: unicodedata.category(char)[0] == "N")
    # Unicode White_Space (used by the pinned tokenizer regex), excluding the
    # extra C0 separators that Python's str.isspace() also treats as spaces.
    whitespace = ranges(
        lambda char: char in "\t\n\v\f\r\x85"
        or unicodedata.category(char) in {"Zs", "Zl", "Zp"}
    )
    decompositions = []
    decomposition_values = []
    compositions = []
    combining_classes = []

    for codepoint in range(MAX_CODEPOINT + 1):
        char = chr(codepoint)
        decomposition = unicodedata.decomposition(char)
        if decomposition and not decomposition.startswith("<"):
            sequence = [int(value, 16) for value in decomposition.split()]
            decompositions.append((codepoint, len(decomposition_values), len(sequence)))
            decomposition_values.extend(sequence)
            if len(sequence) == 2:
                candidate = "".join(chr(value) for value in sequence)
                if unicodedata.normalize("NFC", candidate) == char:
                    compositions.append((sequence[0], sequence[1], codepoint))

        combining_class = unicodedata.combining(char)
        if combining_class:
            combining_classes.append((codepoint, combining_class))

    compositions.sort(key=lambda row: (row[0] << 21) | row[1])

    output = [
        "//! Generated Unicode 15.0.0 tables for Sage's Qwen tokenizer.",
        "//! Regenerate with scripts/generate-qwen-unicode15-tables.py.",
        "pub const UNICODE_DATA_VERSION: &str = \"15.0.0\";\n",
        render_pairs("LETTER_MARK_NUMBER_RANGES", letter_mark_number, "(u32, u32)"),
        render_pairs("LETTER_RANGES", letters, "(u32, u32)"),
        render_pairs("MARK_RANGES", marks, "(u32, u32)"),
        render_pairs("NUMBER_RANGES", numbers, "(u32, u32)"),
        render_pairs("WHITE_SPACE_RANGES", whitespace, "(u32, u32)"),
        render_pairs("CANONICAL_DECOMPOSITION_INDEX", decompositions, "(u32, u32, u8)"),
        "pub const CANONICAL_DECOMPOSITION_VALUES: &[u32] = &[",
    ]
    output.extend(
        "    " + ", ".join(f"0x{value:X}" for value in decomposition_values[index : index + 12]) + ","
        for index in range(0, len(decomposition_values), 12)
    )
    output.extend(["];\n", "pub const COMPOSITION_PAIRS: &[(u64, u32)] = &[\n"])
    output.extend(
        f"    ((0x{left:X}u64 << 21) | 0x{right:X}, 0x{composed:X}),"
        for left, right, composed in compositions
    )
    output.extend(["];\n", "pub const COMBINING_CLASSES: &[(u32, u8)] = &[\n"])
    output.extend(f"    (0x{codepoint:X}, {value})," for codepoint, value in combining_classes)
    output.append("];\n")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    args.output.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
