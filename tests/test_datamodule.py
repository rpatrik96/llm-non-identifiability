import pytest

from llm_non_identifiability.datamodule import GrammarDataModule


@pytest.mark.parametrize(
    "grammar", ["aNbN", "abN", "aNbM", "aNbNaN", "coinflip", "coinflip_mixture"]
)
def test_generate_data_correctly(num_train, num_val, num_test, max_length, grammar):
    data_module = GrammarDataModule(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        max_length=max_length,
        grammar=grammar,
    )
    data_module.prepare_data()

    if grammar == "aNbN":
        num_train = num_val = num_test = max_length // 2
    elif grammar == "aNbNaN":
        num_train = num_val = num_test = max_length // 3

    assert len(data_module.train_dataset) == num_train
    assert len(data_module.val_dataset) == num_val
    assert len(data_module.test_dataset) == num_test

    if grammar not in ["coinflip", "coinflip_mixture"]:
        max_length_offset = 2  # +2 for SOS and EOS tokens
    else:
        max_length_offset = 0

    assert data_module.train_dataset.data.shape[1] == max_length + max_length_offset
    assert data_module.val_dataset.data.shape[1] == max_length + max_length_offset
    assert data_module.test_dataset.data.shape[1] == max_length + max_length_offset
