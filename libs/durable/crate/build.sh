#!/usr/bin/env bash
# Build the kernel to WebAssembly into ../wasm.
#
# Needs wasm-pack and a clang that targets wasm32 (Apple's does), which
# compiles SQLite. The toolchain's llvm-ar archives it: other archivers write
# indexes rust-lld cannot read for wasm objects.
#
# DURABLE_CORE_PATH=~/src/deepagents/libs/durable/crates/core builds against a
# local checkout of the kernel instead of the pinned git revision.
set -euo pipefail
cd "$(dirname "$0")"

host=$(rustc -vV | sed -n 's/^host: //p')
export AR_wasm32_unknown_unknown="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-ar"

cargo_args=()
if [[ -n "${DURABLE_CORE_PATH:-}" ]]; then
  cargo_args+=(--config "patch.\"https://github.com/semistrict/deepagents\".durable-core.path=\"$DURABLE_CORE_PATH\"")
fi

wasm-pack build --target web --weak-refs --out-dir ../wasm --out-name durable --no-pack --release -- ${cargo_args[@]+"${cargo_args[@]}"}
rm -f ../wasm/.gitignore
