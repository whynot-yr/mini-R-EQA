# R-EQA Mini Pipeline

This repository implements a minimal R-EQA-style pipeline for understanding retrieval-augmented episodic memory QA.

## Goal

Build a small and controllable pipeline:

episode captions → retrieval → prompt construction → answer generation

## Current Plan

- [ ] Add sample episode format
- [ ] Add TF-IDF retrieval baseline
- [ ] Add sentence-transformer retrieval
- [ ] Add prompt builder
- [ ] Add simple answer generation script
- [ ] Add experiment logging

## Project Structure

```text
src/        core implementation
scripts/    runnable scripts
configs/    experiment configs
data/       sample data only
notes/      daily logs and design notes
reports/    summarized results
