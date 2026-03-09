import cudaq


def convert(result_dict: dict) -> cudaq.SampleResult:
    sample_result = cudaq.SampleResult()
    sample_result.deserialize(result_dict)
    return sample_result
