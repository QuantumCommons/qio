import cirq


def convert(result_dict: dict, **kwargs) -> "cirq.Result":
    if kwargs:
        result_dict.update(kwargs)

    cirq_result = cirq.ResultDict._from_json_dict_(**result_dict)

    return cirq_result
