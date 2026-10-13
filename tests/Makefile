SDK_DEPS := requirements.txt
PERCEVAL_DEPS := requirements-perceval.txt

.PHONY: install
install:
	pip3 install --upgrade pip
	pip3 install -r $(SDK_DEPS)
	export PYTHONPATH=/src

.PHONY: install-perceval
install-perceval:
	pip3 install --upgrade pip
	pip3 install -r $(PERCEVAL_DEPS)
	export PYTHONPATH=/src

.PHONY: test
test:
	pytest -s --showprogress -vv src/ -k "not perceval"

.PHONY: test-perceval
test-conv-perceval:
	pytest -s --showprogress -vv src/circuit_conversion/ src/result_conversion/ -k "perceval"

.PHONY: test-circuit-conv
test-circuit-conv:
	pytest -s --showprogress -vv src/circuit_conversion/ -k "not perceval"

.PHONY: test-circuit-perceval
test-circuit-conv-perceval:
	pytest -s --showprogress -vv src/circuit_conversion/ -k "perceval"

.PHONY: test-result-conv
test-result-conv:
	pytest -s --showprogress -vv src/result_conversion/ -k "not perceval"

.PHONY: test-result-conv-perceval
test-result-conv-perceval:
	pytest -s --showprogress -vv src/result_conversion/ -k "perceval"
