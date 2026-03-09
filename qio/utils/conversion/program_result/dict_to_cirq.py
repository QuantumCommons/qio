import cirq


def convert(result_dict: dict, **kwargs) -> "cirq.Result":
    kwargs = kwargs or {}

    result_dict.update(kwargs)

    cirq_result = cirq.ResultDict._from_json_dict_(**result_dict)

    return cirq_result
