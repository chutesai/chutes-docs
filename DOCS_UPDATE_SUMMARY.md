# Documentation Update Summary

**Date:** December 2, 2025  
**Branch:** `experimental`  
**Author:** AI-Assisted Review Session

---

## Overview

This document summarizes a comprehensive review and update of the Chutes documentation. The primary goal was to ensure **accuracy, consistency, and alignment** with the latest Chutes SDK, CLI, and platform capabilities.

## 1. Audit Overview

This report summarizes the deep documentation audit performed on the `chutes-docs` repository and the subsequent updates made to align documentation with the codebase (`chutes`, `chutes-api`, `chutes-miner`).

## 2. Audit Findings

The following discrepancies and areas for improvement were identified during the audit:

### Getting Started

- **Installation**: Verified `pip install chutes` is correct.
- **Quickstart**: Found outdated model references (`microsoft/DialoGPT-medium`) which is no longer the recommended starting point. Node selector configurations needed updates to match current platform constraints.

### CLI Reference

- **CLI Commands**: Verified against `chutes/entrypoint/` scripts. The documentation accurately reflects available commands (`build`, `deploy`, `register`, `login`, `logout`, `init`, `status`, `logs`, `delete`, `list`).

### SDK Reference

- **Core Components**: Verified `Chute`, `Image`, `Cord`, and `Job` classes against their Python implementations.
- **NodeSelector**: Checked fields against `chutes/chute/node_selector.py`.
- **Templates**: Verified template functions (`build_vllm_chute`, `build_diffusion_chute`) against `chutes/chute/template/`.

### Examples

- **Outdated Examples**: `llm-chat.md` and `image-generation.md` used older models (Stable Diffusion 1.5, DialoGPT) and required updates to modern equivalents (FLUX.1, Llama 3.2).
- **SDK Usage**: Found incorrect method calls (e.g., `chute.deploy()` instead of CLI usage) in `custom-chute-complete.md`.

### Miner Resources

- **Overview & Ansible**: Documentation aligns with the `chutes-miner` repo structure.
- **Scoring**: Verified against `chutes-api` scoring logic.

## 3. Updates Executed

### Key Fixes

- **Quickstart & Basic Examples**:
  - Replaced deprecated `microsoft/DialoGPT-medium` with `unsloth/Llama-3.2-1B-Instruct` in `quickstart.md` and `llm-chat.md`.
  - Corrected `NodeSelector` configurations to match current GPU requirements.
- **Image Generation**:
  - Rewrote `image-generation.md` to use **FLUX.1 [dev]**.
  - Updated pipeline initialization code to match `diffusers` FluxPipeline requirements.
- **SDK Usage Corrections**:
  - Fixed `custom-chute-complete.md` to remove the incorrect `chute.deploy()` method call, instructing users to use the CLI `chutes deploy` command instead.

### New Content Created

1.  **Reasoning Models Guide (`src/guides/reasoning-models.md`)**

    - Deployment guide for **DeepSeek R1** using the `sglang` template.
    - Covers hardware requirements for full vs. distilled models and handling `<think>` tags.

2.  **Modern Audio Guide (`src/guides/modern-audio.md`)**

    - **TTS**: Guide for deploying **Kokoro-82M**.
    - **STT**: Guide for deploying **Whisper v3**.

3.  **RAG Pipeline Guide (`src/guides/rag-pipeline.md`)**

    - End-to-end tutorial for Retrieval Augmented Generation.
    - Covers modular deployment of Embedding, Vector DB (Chroma), and Generation chutes.

4.  **Miner Maintenance (`src/guides/miner-maintenance.md`)**

    - Operational guide for node operators covering chart updates, disk cleanup, and K3s/GPU troubleshooting.

5.  **Production Readiness (`src/guides/production-readiness.md`)**

    - Checklist for moving to production: security (scoped keys), reliability (health checks), and scaling configuration.

6.  **Agents and Tool Use (`src/guides/agents-and-tools.md`)**
    - Guide for function calling, tool use, and building agents with vLLM/SGLang.
    - Complete Python client loop for agent execution.
    - Structured JSON output (JSON mode) and SGLang constrained generation.

## Methodology (Previous Pass)

1.  **Base Code Comparison:** Each documentation folder was systematically compared against the actual Chutes SDK source code (`chutes-ai/chutes` repository) to verify API signatures, default values, and available options.
2.  **Dependency Modernization:** All code examples were updated to use current, stable versions of key libraries (PyTorch 2.4+, Transformers 4.44+, CUDA 12.x base images).
3.  **Consistency Pass:** Ensured uniform patterns across all examples, including:
    - `NodeSelector` passed as a keyword argument.
    - Correct `Image` class method chaining.
    - Accurate CLI command syntax.
4.  **Consolidation:** Merged redundant content and removed stale files.
5.  **Gap Analysis:** Identified and filled documentation gaps with new, high-value content.

## Summary of Changes by Folder

### `src/guides/`

| File                          | Change Type      | Summary                                                                                                                                |
| ----------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `performance-optimization.md` | **DELETED**      | Content merged into `performance.md` to eliminate redundancy.                                                                          |
| `performance.md`              | **MAJOR UPDATE** | Consolidated all performance content. Streamlined from ~800 lines to ~400. Focused on actionable GPU, memory, and batching strategies. |
| `best-practices.md`           | **UPDATE**       | Fixed internal links. Removed duplicated content now in `performance.md`.                                                              |
| `agents-and-tools.md`         | **NEW**          | Added guide for function calling, tool use, and building agents with vLLM/SGLang.                                                      |
| `reasoning-models.md`         | **NEW**          | DeepSeek R1 deployment guide.                                                                                                          |
| `modern-audio.md`             | **NEW**          | Kokoro TTS and Whisper v3 STT guide.                                                                                                   |
| `rag-pipeline.md`             | **NEW**          | End-to-end RAG tutorial.                                                                                                               |
| `miner-maintenance.md`        | **NEW**          | Operational guide for miners.                                                                                                          |
| `production-readiness.md`     | **NEW**          | Checklist for production deployments.                                                                                                  |

### `src/examples/`

| File                       | Change Type      | Summary                                                                                           |
| -------------------------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| `streaming.md`             | **DELETED**      | Was a redirect stub; content already in `streaming-responses.md`.                                 |
| `simple-text-analysis.md`  | UPDATE           | Modernized dependencies, fixed `NodeSelector` usage, corrected code block formatting.             |
| `streaming-responses.md`   | UPDATE           | Updated base image to CUDA 12.4, fixed `Chute` definition.                                        |
| `multi-model-analysis.md`  | UPDATE           | Fixed `image` variable reference, added missing `import time`.                                    |
| `image-generation.md`      | **MAJOR UPDATE** | Rewrote to use FLUX.1 [dev], added missing `import base64`, fixed `Response` import from FastAPI. |
| `audio-processing.md`      | UPDATE           | Modernized to CUDA 12.1, updated Whisper/PyTorch versions.                                        |
| `embeddings.md`            | UPDATE           | Updated image definition, fixed `NodeSelector` as keyword arg.                                    |
| `video-generation.md`      | MAJOR UPDATE     | Switched primary example to Wan2.1-14B with distributed multi-GPU inference.                      |
| `custom-training.md`       | UPDATE           | Updated base image and dependencies, corrected `NodeSelector` usage.                              |
| `llm-chat.md`              | **UPDATE**       | Updated model to Llama 3.2 1B Instruct.                                                           |
| `custom-chute-complete.md` | **UPDATE**       | Fixed SDK usage (CLI deploy vs `chute.deploy`).                                                   |

### `src/help/`

| File                 | Change Type | Summary                                                                                            |
| -------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| `faq.md`             | UPDATE      | Corrected CLI commands (e.g., `chutes auth login` → `chutes register`). Updated template examples. |
| `troubleshooting.md` | UPDATE      | Modernized dependency versions in solutions. Updated GPU OOM fixes with `NodeSelector` patterns.   |

### `src/cli/`

| File          | Change Type | Summary                                                        |
| ------------- | ----------- | -------------------------------------------------------------- |
| `overview.md` | VERIFIED    | Confirmed command structure matches base code.                 |
| `account.md`  | VERIFIED    | Confirmed `register`, `keys`, `secrets` commands.              |
| `build.md`    | UPDATE      | Updated base image recommendation to `parachutes/python:3.12`. |
| `deploy.md`   | VERIFIED    | Confirmed `--accept-fee` flag and `NodeSelector` options.      |
| `manage.md`   | VERIFIED    | Confirmed `chutes`, `images`, `share`, `warmup` commands.      |

### `src/sdk-reference/`

| File               | Change Type | Summary                                                                                                  |
| ------------------ | ----------- | -------------------------------------------------------------------------------------------------------- |
| `chute.md`         | UPDATE      | Updated `allow_external_egress` default (False), added `tee` parameter, documented lifecycle decorators. |
| `cord.md`          | UPDATE      | Verified `passthrough`, `stream` parameters. Clarified `input_schema` vs `minimal_input_schema`.         |
| `image.md`         | UPDATE      | Updated recommended base image. Clarified `apt_install` vs `run_command`. Added `with_maintainer`.       |
| `job.md`           | VERIFIED    | Confirmed `upload`, `ssh`, `timeout` parameters.                                                         |
| `node-selector.md` | UPDATE      | Added `h200`, `mi300x`, `l40` GPUs. Refined VRAM guidance.                                               |
| `templates.md`     | UPDATE      | Added `build_sglang_chute`, `build_diffusion_chute`, `build_embedding_chute` documentation.              |

### `src/getting-started/`

| File                 | Change Type | Summary                                                     |
| -------------------- | ----------- | ----------------------------------------------------------- |
| `installation.md`    | UPDATE      | Streamlined installation steps.                             |
| `first-chute.md`     | UPDATE      | Removed redundant content, focused on core workflow.        |
| `running-a-chute.md` | UPDATE      | Simplified examples, removed duplication with other guides. |
| `authentication.md`  | UPDATE      | Minor link fix.                                             |
| `quickstart.md`      | **UPDATE**  | Updated to Llama 3.2 1B Instruct.                           |

### `src/miner-resources/`

| File          | Change Type | Summary                                         |
| ------------- | ----------- | ----------------------------------------------- |
| `overview.md` | UPDATE      | Streamlined content, removed outdated sections. |
| `ansible.md`  | UPDATE      | Simplified playbook examples.                   |

## Major Highlights

### 1. New Guides & Content

- **Agents and Tool Use**: Net-new guide covering function calling, tool use, and building agents.
- **Reasoning Models**: Dedicated guide for DeepSeek R1.
- **Modern Audio**: Kokoro and Whisper v3 examples.
- **RAG Pipeline**: Modular deployment guide for RAG.
- **Miner Operations**: "Day 2" maintenance guide.
- **Production Readiness**: Checklist for reliable deployments.

### 2. Consolidated Performance Guide

Merged `performance-optimization.md` into `performance.md`, reducing ~1200 combined lines to ~400 focused, actionable lines. Eliminated redundancy and improved scannability.

### 3. Modernized All Code Examples

Every code example now uses:

- **Base Image:** `nvidia/cuda:12.1-devel-ubuntu22.04` or `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- **PyTorch:** `>=2.4.0`
- **Transformers:** `>=4.44.0`
- **Correct SDK patterns:** `NodeSelector` as keyword argument, proper `Image` chaining.

### 4. Verified CLI & SDK Against Source

All CLI and SDK documentation was compared against `chutes-ai/chutes/` source files. Commands, options, defaults, constructor parameters, and decorators are now accurate.

## 4. Future Recommendations & Next Steps

### For Documentation Maintainers

1.  **Automated Doc Testing**: Implement a system to test code snippets in documentation against the SDK to catch regressions.
2.  **Versioned Docs**: Ensure documentation is versioned alongside the SDK to prevent mismatches.
3.  **More Examples**: Add examples for `tei` (Text Embeddings Inference) templates if they become more prominent.
4.  **Miner Setup Guide**: Consider creating a dedicated "Miner Quickstart" script or interactive guide to simplify the complex Ansible/K8s setup.
5.  **Review Formatting**: Check that the new markdown files render correctly in the documentation site generator.
