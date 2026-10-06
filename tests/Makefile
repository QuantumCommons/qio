SDK_DEPS := requirements.txt

.PHONY: install
install:
	pip3 install --upgrade pip
	pip3 install -r $(SDK_DEPS)
	export PYTHONPATH=/src

.PHONY: test
test:
	pytest -s --showprogress -vv src/

.PHONY: test-circuit-conv
test-circuit-conv:
	pytest -s --showprogress -vv src/circuit_conversion/
