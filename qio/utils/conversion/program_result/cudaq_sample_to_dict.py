from cudaq import SampleResult


def convert(sample_result: SampleResult) -> dict:
    return sample_result.serialize()
