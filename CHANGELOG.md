# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Model pricing tables for OpenAI chat + embedding families with per-response and per-chunk cost computation.
- API + UI support for selecting system messages, embedding models, and viewing token/cost breakdowns.
- Token counting utility backed by tiktoken with fallback heuristics.
- `tiktoken` dependency and pricing utilities module.

## [0.1.0] - 2025-09-20
### Added
- Initial release of Semantic Landscape Sampler with backend FastAPI service, frontend React visualiser, embeddings pipeline, clustering, and interactive point cloud.
