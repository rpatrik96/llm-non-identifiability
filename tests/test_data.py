from llm_non_identifiability.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_aN_bM_grammar_data,
    pad,
    PAD,
)

import numpy as np


def test_aNbN_grammar_equal_as_bs(n, max_length):
    sequences = generate_aNbN_grammar_data(n, max_length)
    for sequence in sequences:
        num_as = np.sum(sequence == 0)
        num_bs = np.sum(sequence == 1)
        assert num_as == num_bs


def test_aNbN_grammar_as_before_bs(n, max_length):
    sequences = generate_aNbN_grammar_data(n, max_length)
    for sequence in sequences:
        # find the first b
        first_b = np.where(sequence == 1)[0][0]
        # find the last a
        last_a = np.where(sequence == 0)[0][-1]
        assert first_b > last_a


def test_abN_equal_as_bs(n, max_length):
    data = generate_abN_grammar_data(n, max_length)
    for sequence in data:
        num_a = np.sum(sequence == 0)
        num_b = np.sum(sequence == 1)
        assert num_a == num_b


def test_aNbM_grammar_as_before_bs(n, max_length):
    sequences = generate_aN_bM_grammar_data(n, max_length)
    for sequence in sequences:
        # find the first b
        first_b = np.where(sequence == 1)[0][0]
        # find the last a
        last_a = np.where(sequence == 0)[0][-1]
        assert first_b > last_a


def test_pad_varying_sequence_lengths():
    data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    expected_result = [[1, 2, 3, PAD], [4, 5, PAD, PAD], [6, 7, 8, 9]]
    result = pad(data)

    # check that the result is as expected with .all()
    assert (np.array(result) == np.array(expected_result)).all()
