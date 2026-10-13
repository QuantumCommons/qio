SDK_DEPS := requirements.txt

.PHONY: install
install:
	pip3 install --upgrade pip
	pip3 install -r $(SDK_DEPS)
	export PYTHONPATH=/src

# mimiqcircuits and Perceval both register a `circuit.proto` into protobuf's
# global descriptor pool, so they cannot be imported in the same Python process.
# The photonic (Perceval) batteries (circuit + result conversions) therefore run
# in their own pytest invocation:
#   - the main invocation runs everything "-k not perceval" (mimiq + others, no
#     Perceval ever imported),
#   - a second invocation runs only the "perceval" tests (no mimiq imported).
.PHONY: test
test:
	pytest -s --showprogress -vv src/ -k "not perceval"
	pytest -s --showprogress -vv src/circuit_conversion/ src/result_conversion/ -k "perceval"

.PHONY: test-circuit-conv
test-circuit-conv:
	pytest -s --showprogress -vv src/circuit_conversion/ -k "not perceval"
	pytest -s --showprogress -vv src/circuit_conversion/ -k "perceval"

.PHONY: test-photonic
test-photonic:
	pytest -s --showprogress -vv src/circuit_conversion/ -k "perceval"

.PHONY: test-result-conv
test-result-conv:
	pytest -s --showprogress -vv src/result_conversion/ -k "not perceval"
	pytest -s --showprogress -vv src/result_conversion/ -k "perceval"
