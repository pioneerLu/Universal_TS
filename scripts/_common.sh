#!/usr/bin/env bash

_find_root() {
    local d
    d="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    while [ "$d" != "/" ]; do
        if [ -f "$d/train.py" ]; then
            echo "$d"
            return 0
        fi
        d="$(dirname "$d")"
    done
    echo "Cannot locate train.py" >&2
    return 1
}

ROOT="$(_find_root)"
cd "$ROOT"
echo "ROOT=$ROOT"
