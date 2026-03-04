# Makefile (repo root)

SHELL := /bin/bash

# Default config paths (override like: make preprocess CONFIG=configs/preprocess.yaml)
CONFIG_PREPROCESS ?= configs/preprocess.yaml
CONFIG_TRAIN      ?= configs/train.yaml
CONFIG_EVAL       ?= configs/eval.yaml
CONFIG_SERVE      ?= configs/serve.yaml

.PHONY: help preprocess train eval serve stress

help:
	@echo ""
	@echo "Targets:"
	@echo "  make preprocess   Run data preprocessing (uses $(CONFIG_PREPROCESS))"
	@echo "  make train        Run training (uses $(CONFIG_TRAIN))"
	@echo "  make eval         Run evaluation (uses $(CONFIG_EVAL))"
	@echo "  make serve        Run serving (uses $(CONFIG_SERVE))"
	@echo "  make stress       Run stress test (define later)"
	@echo ""
	@echo "Override example:"
	@echo "  make preprocess CONFIG_PREPROCESS=path/to/other.yaml"
	@echo ""

preprocess:
	@echo "==> Preprocess using config: $(CONFIG_PREPROCESS)"
	python dataset_pipeline/pipeline.py --config "$(CONFIG_PREPROCESS)"

train:
	@echo "==> Train using config: $(CONFIG_TRAIN)"
	python main/train.py --config "$(CONFIG_TRAIN)"

eval:
	@echo "==> Eval using config: $(CONFIG_EVAL)"
	python main/eval.py --config "$(CONFIG_EVAL)"

serve:
	@echo "==> Serve using config: $(CONFIG_SERVE)"
	python main/serve.py --config "$(CONFIG_SERVE)"

stress:
	@echo "==> Stress test (not implemented yet)"
	@exit 1