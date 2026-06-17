#!/bin/sh
set -e

# Ensure sample data and model checkpoint exist
if [ ! -f "data/samples/athens_owl.json" ]; then
  mint-atlas generate --mint-id athens_owl
fi
if [ ! -f "data/samples/seleucid_antioch.json" ]; then
  mint-atlas generate --mint-id seleucid_antioch --seed 7
fi
if [ ! -f "checkpoints/mint_atlas_gnn.pt" ]; then
  mint-atlas train --epochs 40
fi

exec mint-atlas serve --host 0.0.0.0 --port "${PORT:-8000}"
