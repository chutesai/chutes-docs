# Documentation Review List

## New Pages (Please Review)

### Guides
1. **`src/guides/reasoning-models.md`**
   - Guide for deploying DeepSeek R1 reasoning models
   - Covers both distilled (8B) and full (671B) versions
   - Includes hardware requirements and usage patterns

2. **`src/guides/modern-audio.md`**
   - Modern audio processing with Kokoro-82M (TTS) and Whisper v3 (STT)
   - Complete deployment examples with voice packs
   - Base64 audio handling

3. **`src/guides/rag-pipeline.md`**
   - End-to-end Retrieval Augmented Generation tutorial
   - Modular architecture: Embeddings → ChromaDB → LLM
   - Client orchestration example

4. **`src/guides/production-readiness.md`**
   - Production deployment checklist
   - Security, scaling, and reliability best practices
   - Scoped API keys and monitoring

### Miner Resources
5. **`src/miner-resources/miner-maintenance.md`**
   - "Day 2" operations guide for miners
   - Routine maintenance, troubleshooting K3s/GPU issues
   - Safe node reboot procedures

---

## Updated Pages (Please Review)

### Getting Started
1. **`src/getting-started/quickstart.md`**
   - Updated model: `microsoft/DialoGPT-medium` → `unsloth/Llama-3.2-1B-Instruct`
   - Fixed NodeSelector configuration

### Examples
2. **`src/examples/llm-chat.md`**
   - Updated VLLM example to use Llama 3.2 1B Instruct
   - Added production example with Gemma 3 1B

3. **`src/examples/image-generation.md`**
   - Complete rewrite: Stable Diffusion → FLUX.1 [dev]
   - Updated to use FluxPipeline from diffusers
   - Corrected NodeSelector for 80GB VRAM requirement

4. **`src/examples/custom-chute-complete.md`**
   - Fixed incorrect SDK usage: removed `chute.deploy()` method call
   - Updated to use CLI: `chutes deploy` command

---

## Files Changed (Git Stats)
- **11 files changed**
- **1,005 insertions (+)**
- **168 deletions (-)**

## Commit Hash
`49f54d4`

## Branch
`experimental`

---

## Review Priorities

### High Priority (Core User Journey)
- [ ] `src/getting-started/quickstart.md` - First thing new users see
- [ ] `src/examples/llm-chat.md` - Most common use case
- [ ] `src/examples/image-generation.md` - Second most common use case

### Medium Priority (Advanced Features)
- [ ] `src/guides/reasoning-models.md` - Emerging use case (DeepSeek R1)
- [ ] `src/guides/rag-pipeline.md` - Popular enterprise pattern
- [ ] `src/guides/production-readiness.md` - Critical for serious users

### Low Priority (Specialized)
- [ ] `src/guides/modern-audio.md` - Niche but high-quality
- [ ] `src/miner-resources/miner-maintenance.md` - For miners only

