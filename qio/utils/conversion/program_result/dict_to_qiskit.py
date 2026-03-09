from qiskit.result import Result


def _hex_to_bitstring(value, nb_bits):
    mask = (1 << nb_bits) - 1
    return format(value & mask, f"0{nb_bits}b")


def convert(result_dict: dict, **kwargs) -> Result:
    results = result_dict["results"]

    for experiment in results:
        exp_data = experiment.get("data", {})
        counts = exp_data.get("counts", None)
        n_qubits = experiment["header"]["n_qubits"]

        if counts:
            new_counts = {}

            for bitstring, count in counts.items():
                if bitstring.startswith("0x"):
                    nbits = _hex_to_bitstring(int(bitstring, 16), n_qubits)
                else:
                    nbits = bitstring

                new_counts[nbits] = count

            exp_data["counts"] = new_counts

    data = {
        "results": results,
        "success": result_dict["success"],
        "header": result_dict.get("header"),
        "metadata": result_dict.get("metadata"),
    }

    if kwargs:
        data.update(kwargs)

    return Result.from_dict(data)
