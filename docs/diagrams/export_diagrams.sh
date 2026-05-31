#!/usr/bin/env bash
# Export Mermaid (.mmd) diagrams to PNG and SVG
#
# Requires (one of):
#   a) mmdc + puppeteer (local) — run via mmdc directly
#   b) curl + jq + kroki.io    — fallback web API (no local deps)
#
# Install mmdc:  npm install -g @mermaid-js/mermaid-cli

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="${1:-$SCRIPT_DIR}"
OUT_DIR="${2:-$INPUT_DIR/out}"

mkdir -p "$OUT_DIR"

render_mmdc() {
  local f="$1" out="$2"
  local cfg="$SCRIPT_DIR/puppeteer-config.json"
  if [ -f "$cfg" ]; then
    mmdc -i "$f" -o "$out" -p "$cfg"
  else
    mmdc -i "$f" -o "$out"
  fi
}

render_kroki() {
  local f="$1" out="$2" fmt="$3"
  local mime="png"
  [ "$fmt" = "svg" ] && mime="svg"
  curl -sf -X POST "https://kroki.io/mermaid/$mime" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"diagram_source":%s,"diagram_type":"mermaid"}' "$(cat "$f" | jq -Rs .)")" \
    -o "$out"
}

try_render() {
  local f="$1" out="$2" fmt="$3"
  if command -v mmdc &>/dev/null; then
    if render_mmdc "$f" "$out" 2>/dev/null; then
      return 0
    fi
    echo "  mmdc failed, trying fallback ..."
  fi
  if command -v curl &>/dev/null && command -v jq &>/dev/null; then
    render_kroki "$f" "$out" "$fmt" && return 0
    echo "  kroki fallback failed."
  else
    echo "  fallback requires curl + jq."
  fi
  return 1
}

for mmd in "$INPUT_DIR"/*.mmd; do
  [ -f "$mmd" ] || continue
  base="$(basename "$mmd" .mmd)"

  for fmt in png svg; do
    out="$OUT_DIR/$base.$fmt"
    echo "Exporting $base.$fmt ..."
    if ! try_render "$mmd" "$out" "$fmt"; then
      echo "  SKIPPED $base.$fmt (no renderer available)"
    fi
  done
done

echo "Done. Outputs in: $OUT_DIR"
