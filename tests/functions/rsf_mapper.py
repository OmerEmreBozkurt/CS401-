#!/usr/bin/env python3
"""
RSF File Name to Alphanumeric ID Mapper - Fixed Version
- Generates IDs using only letters and numbers (no punctuation/symbols).
- Starts with 1-char IDs, then 2-char, 3-char... as needed using a base62 alphabet.
- Stable mapping: most frequent filenames get the shortest IDs; ties broken lexicographically.
"""

import sys
import os
from collections import Counter
from typing import Dict, List, Iterable, Tuple
import itertools

BASE62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def base62_ids() -> Iterable[str]:
    """Yield an infinite sequence of base62 IDs: a..zA..Z0..9, aa..a9.., ab.. etc."""
    # 1-char
    for c in BASE62:
        yield c
    # n-char for n >= 2
    length = 2
    while True:
        for tup in itertools.product(BASE62, repeat=length):
            yield "".join(tup)
        length += 1


def parse_rsf_lines(lines: Iterable[str]) -> List[Tuple[str, List[str]]]:
    """
    Parse RSF-like lines.
    Each line is tokenized by whitespace. We assume the filenames start from token 1 onwards.
    Example supported patterns:
      contain <cluster> <file>
      include <src> <dst>
      depends <src> <dst1> <dst2> ...
      <rel> <arg1> <arg2> ...
    Returns list of (relation, args[]) where args contain file-like tokens.
    """
    parsed = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rel, args = parts[0], parts[1:]
        parsed.append((rel, args))
    return parsed


def collect_filenames(parsed: List[Tuple[str, List[str]]]) -> Counter:
    """Collect frequency of all filename tokens from parsed RSF tuples."""
    freq = Counter()
    for _, args in parsed:
        for token in args:
            freq[token] += 1
    return freq


def build_alnum_mapping(freq: Counter) -> Dict[str, str]:
    """
    Build a deterministic alphanumeric mapping:
    - Sort by (-count, token) to prioritize frequent and stable ordering.
    - Assign shortest available base62 IDs first.
    """
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    id_gen = base62_ids()
    mapping: Dict[str, str] = {}
    for filename, _ in items:
        mapping[filename] = next(id_gen)
    return mapping


def rewrite_lines(parsed: List[Tuple[str, List[str]]], mapping: Dict[str, str]) -> List[str]:
    """
    Rewrite lines with mapped tokens.
    Special handling for 'depends' relation - outputs without the relation name.
    """
    out_lines: List[str] = []
    for rel, args in parsed:
        mapped_args = [mapping.get(tok, tok) for tok in args]
        if rel == "depends":
            # For "depends", output only mapped args joined by space
            out_lines.append(" ".join(mapped_args))
        else:
            # For other relations, include the relation name
            out_lines.append(" ".join([rel] + mapped_args))
    return out_lines


def save_mapping(mapping: Dict[str, str], path: str) -> None:
    """Write mapping as 'filename -> ID' per line."""
    with open(path, "w", encoding="utf-8") as f:
        for filename, tok in sorted(mapping.items()):
            f.write(f"{filename} -> {tok}\n")


def save_reverse_mapping(mapping: Dict[str, str], path: str) -> None:
    """Write reverse mapping as 'ID -> filename' per line."""
    with open(path, "w", encoding="utf-8") as f:
        reverse = {v: k for k, v in mapping.items()}
        for tok in sorted(reverse.keys(), key=lambda x: (len(x), x)):
            f.write(f"{tok} -> {reverse[tok]}\n")


def load_file(path: str) -> List[str]:
    """Load file and return lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{path}': {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python rsf_mapper.py <input.rsf> [mapping.txt] [output.rsf]")
        print(" - mapping.txt defaults to 'mapping.txt' in the same directory as input")
        print(" - output.rsf defaults to '<basename>_compact.rsf'")
        print("\nExample:")
        print("  python rsf_mapper.py dependencies.rsf")
        print("  python rsf_mapper.py dependencies.rsf my_mapping.txt compact_deps.rsf")
        sys.exit(1)

    in_path = sys.argv[1]

    # Set default paths
    base_dir = os.path.dirname(in_path) or '.'
    base_name = os.path.splitext(os.path.basename(in_path))[0]

    map_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, "mapping.txt")
    reverse_map_path = os.path.splitext(map_path)[0] + "_reverse.txt"
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_dir, f"{base_name}_compact.rsf")

    print(f"Loading input file: {in_path}")
    lines = load_file(in_path)

    print(f"Parsing {len(lines)} lines...")
    parsed = parse_rsf_lines(lines)

    if not parsed:
        print("Warning: No valid lines found in input file!")
        sys.exit(1)

    print(f"Collecting filename frequencies...")
    freq = collect_filenames(parsed)

    if not freq:
        print("Warning: No filenames found in parsed data!")
        sys.exit(1)

    print(f"Building mapping for {len(freq)} unique tokens...")
    mapping = build_alnum_mapping(freq)

    # Save mappings
    save_mapping(mapping, map_path)
    save_reverse_mapping(mapping, reverse_map_path)

    # Rewrite lines
    print("Rewriting lines with compact IDs...")
    compact_lines = rewrite_lines(parsed, mapping)

    # Save compact output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(compact_lines))

    # Print statistics
    print("\n" + "=" * 60)
    print(f"Unique tokens: {len(freq)}")
    for token, count in freq.most_common(5):
        print(f"  '{token}' -> '{mapping[token]}' (appears {count} times)")
    print(f"\nOutputs:")
    print(f"  Mapping: {map_path}")
    print(f"  Reverse mapping: {reverse_map_path}")
    print(f"  Compact RSF: {out_path}")


if __name__ == "__main__":
    main()