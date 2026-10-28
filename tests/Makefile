PYTHON ?= python3.12

GATE_DEPS := requirements-gate.txt
GATE_VENV := .env-gate
ANALOG_DEPS := requirements-analog.txt
ANALOG_VENV := .env-analog
PHOTONIC_DEPS := requirements-photonic.txt
PHOTONIC_VENV := .env-photonic
REPORT_DEPS := requirements-report.txt
REPORT_VENV := .env-report

REPO_ROOT := $(abspath $(CURDIR)/..)
export PYTHONPATH := $(REPO_ROOT)

.PHONY: install
install: install-gate install-analog install-photonic install-report

$(GATE_VENV)/bin/python:
	$(PYTHON) -m venv $(GATE_VENV)

$(ANALOG_VENV)/bin/python:
	$(PYTHON) -m venv $(ANALOG_VENV)

$(PHOTONIC_VENV)/bin/python:
	$(PYTHON) -m venv $(PHOTONIC_VENV)

$(REPORT_VENV)/bin/python:
	$(PYTHON) -m venv $(REPORT_VENV)

$(GATE_VENV)/.installed: requirements-gate.txt | $(GATE_VENV)/bin/python
	$(GATE_VENV)/bin/python -m pip install --upgrade pip
	$(GATE_VENV)/bin/python -m pip install -r $(GATE_DEPS)
	@touch $@

$(ANALOG_VENV)/.installed: requirements-analog.txt | $(ANALOG_VENV)/bin/python
	$(ANALOG_VENV)/bin/python -m pip install --upgrade pip
	$(ANALOG_VENV)/bin/python -m pip install -r $(ANALOG_DEPS)
	@touch $@

$(PHOTONIC_VENV)/.installed: requirements-photonic.txt | $(PHOTONIC_VENV)/bin/python
	$(PHOTONIC_VENV)/bin/python -m pip install --upgrade pip
	$(PHOTONIC_VENV)/bin/python -m pip install -r $(PHOTONIC_DEPS)
	@touch $@

$(REPORT_VENV)/.installed: requirements-report.txt | $(REPORT_VENV)/bin/python
	$(REPORT_VENV)/bin/python -m pip install --upgrade pip
	$(REPORT_VENV)/bin/python -m pip install -r $(REPORT_DEPS)
	@touch $@

.PHONY: install-gate
install-gate: $(GATE_VENV)/.installed

.PHONY: install-analog
install-analog: $(ANALOG_VENV)/.installed

.PHONY: install-photonic
install-photonic: $(PHOTONIC_VENV)/.installed

.PHONY: install-report
install-report: $(REPORT_VENV)/.installed

.PHONY: test
test: test-gate test-analog test-photonic

.PHONY: test-gate
test-gate: install-gate
	$(GATE_VENV)/bin/python -m pytest -s --showprogress -vv src/ -m gate

.PHONY: test-analog
test-analog: install-analog
	$(ANALOG_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ src/result_conversion/ -m analog

.PHONY: test-photonic
test-photonic: install-photonic
	$(PHOTONIC_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ src/result_conversion/ -m photonic

.PHONY: test-circuit-conv-gate
test-circuit-conv-gate:
	$(GATE_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ -m gate

.PHONY: test-circuit-conv-analog
test-circuit-conv-analog:
	$(ANALOG_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ -m analog

.PHONY: test-circuit-conv-photonic
test-circuit-conv-photonic:
	$(PHOTONIC_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ -m photonic

.PHONY: test-result-conv-gate
test-result-conv-gate:
	$(GATE_VENV)/bin/python -m pytest -s --showprogress -vv src/result_conversion/ -m gate

.PHONY: test-result-conv-analog
test-result-conv-analog:
	$(ANALOG_VENV)/bin/python -m pytest -s --showprogress -vv src/result_conversion/ -m analog

.PHONY: test-result-conv-photonic
test-result-conv-photonic:
	$(PHOTONIC_VENV)/bin/python -m pytest -s --showprogress -vv src/result_conversion/ -m photonic

.PHONY: report
report: install-report
	$(REPORT_VENV)/bin/python tools/generate_report.py

.PHONY: report-observed
report-observed: install-gate install-analog install-photonic install-report
	QIO_LOSS_REPORT_JSON=/tmp/loss-gate.json $(GATE_VENV)/bin/python -m pytest -s --showprogress -vv src/ -m gate
	QIO_LOSS_REPORT_JSON=/tmp/loss-analog.json $(ANALOG_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ src/result_conversion/ -m analog
	QIO_LOSS_REPORT_JSON=/tmp/loss-photonic.json $(PHOTONIC_VENV)/bin/python -m pytest -s --showprogress -vv src/circuit_conversion/ src/result_conversion/ -m photonic
	$(REPORT_VENV)/bin/python tools/generate_report.py \
		--observed /tmp/loss-gate.json \
		--observed /tmp/loss-analog.json \
		--observed /tmp/loss-photonic.json
