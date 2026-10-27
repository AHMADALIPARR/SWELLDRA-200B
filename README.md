[![License: SNAPKITTY OPAQUE SOURCE LICENSE v1.0](https://img.shields.io/badge/License-SNAPKITTY%20OPAQUE%20SOURCE%20LICENSE%20v1.0-blue.svg)](LICENSE)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](requirements.txt)
[![Architecture: 128-Expert MoE](https://img.shields.io/badge/Architecture-128--Expert%20MoE-green.svg)](swelldra/modeling/)

# SWELLDRA — Operation SWELLING: 1M → 200B MoE

**Objective:** Evolve SWELLDRA from 1M to 200B total parameters with sparse MoE activation.

**Total Params:** 200.2B | **Active per token:** ~11.2B | **Experts:** 128 | **Expansion:** 200,000x

## Architecture
- Shared Backbone: d_model=8192, 64 layers, GQA 64Q/8KV, RoPE, RMSNorm, SwiGLU — 8.2B params
- Expert Pool: 128 experts x 1.5B = 192B
  - Language: 40 experts
  - OCR/Doc: 32 experts  
  - SQL: 32 experts
  - Graph: 16 experts
  - Logic: 8 experts
- Router: Top-2 gating with load balancing auxiliary loss
- Verification Layer: SQL schema check, graph invariants, Z3 proof verification

## Quickstart
pip install -r requirements.txt
python run.py --config configs/moe_1m.yaml --mode test

## Capabilities
1. OCR & Document Intelligence (ViT + Donut + TableFormer)
2. SQL & Database Reasoning (schema encoder + execution feedback)
3. Graph Intelligence (entity resolution + multi-hop)
4. Symbolic Logic (FOL + Z3)

## Training Ladder
1M → 100M → 1B → 7B → 45B → 200B
