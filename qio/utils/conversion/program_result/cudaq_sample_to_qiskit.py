import qiskit

def _long_to_bitstring(val: int, size: int) -> str:
    """
    Equivalent to the C++ longToBitString.
    Converts an integer to a binary string padded to the correct bit length.
    """
    # Format to binary, remove '0b' prefix, and pad with leading zeros
    return bin(val)[2:].zfill(size)

def _extract_name(data: List[int], stride: int) -> Tuple[str, int]:
    """Extracts a string (register name) from the data array."""
    n_chars = data[stride]
    stride += 1
    name = "".join(chr(data[i]) for i in range(stride, stride + n_chars))
    stride += n_chars

    return name, stride

def _deserialize_to_dict(data: List[int]) -> Dict[str, Dict[str, int]]:
    """
    Parses the integer array into a dictionary of registers and their counts.
    Matches the logic of ExecutionResult::deserialize and deserializeCounts.
    """
    stride = 0
    all_results = {}

    while stride < len(data):
        # 1. Extract Register Name
        name, stride = _extract_name(data, stride)

        # 2. Extract Counts (deserializeCounts logic)
        num_bitstrings = data[stride]
        stride += 1

        local_counts = {}
        memory = []
        # Each entry is a triplet: [packed_value, bit_size, count]
        for _ in range(num_bitstrings):
            bitstring_as_long = data[stride]
            size_of_bitstring = data[stride + 1]
            count = data[stride + 2]

            bs = _long_to_bitstring(bitstring_as_long, size_of_bitstring)
            local_counts[bs] = count
            memory.extend([bs] * count)
            stride += 3

        all_results[name] = local_counts

    return all_results, memory

def convert() -> :
    parsed_data, memory = __deserialize_to_dict(serialization)
    experiment_results = []

    for reg_name, counts in parsed_data.items():
        shots = sum(counts.values())

        # Encapsulate data in Qiskit's expected format
        data_payload = ExperimentResultData(counts=counts, memory=memory)

        exp_res = ExperimentResult(
            shots=shots,
            success=True,
            data=data_payload,
            header={"name": reg_name, "memory": True},
            status="Done",
        )
        experiment_results.append(exp_res)

    return Result(results=experiment_results, **kwargs)