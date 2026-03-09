import cirq


def convert(cirq_result: cirq.Result, **kwargs) -> dict:
    return cirq_result._json_dict_()
