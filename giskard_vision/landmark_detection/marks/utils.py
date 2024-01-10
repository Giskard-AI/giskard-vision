import numpy as np

# See https://ibug.doc.ic.ac.uk/resources/300-W/ for definition
LEFT_EYE_LEFT_LANDMARK = 36
RIGHT_EYE_RIGHT_LANDMARK = 45


def compute_d_outers(marks):
    return np.sqrt(
        np.einsum("ij->i", (marks[:, LEFT_EYE_LEFT_LANDMARK, :] - marks[:, RIGHT_EYE_RIGHT_LANDMARK, :]) ** 2)
    )
