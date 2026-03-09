from qiskit.result import Result


def convert(qiskit_result: Result) -> dict:
    return qiskit_result.to_dict()
