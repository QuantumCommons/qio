from qiskit_aer.noise import NoiseModel


def convert(noise_model: NoiseModel) -> dict:
    noise_model_dict = noise_model.to_dict(False)

    return noise_model_dict
