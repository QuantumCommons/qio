import cirq


def convert(cirq_result: dict, **kwargs) -> dict:
    return cirq_result._json_dict_()
