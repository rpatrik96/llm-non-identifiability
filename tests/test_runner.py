from pytorch_lightning.trainer import Trainer

from llm_non_identifiability.datamodule import GrammarDataModule
from llm_non_identifiability.runner import LightningGrammarModule
from llm_non_identifiability.data import EOS_token

import torch


def test_fit_and_predict(num_train, num_val, num_test):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule()
    dm = GrammarDataModule(num_train=num_train, num_val=num_val, num_test=num_test)
    trainer.fit(runner, datamodule=dm)

    trainer.predict(runner, datamodule=dm)


def test_predict_inner(max_length, device):
    runner = LightningGrammarModule()

    # Here we test some examples to observe how the model predicts
    examples = [
        torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1, EOS_token.item()]],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [[0, 0, 0, 1, 1, 1, EOS_token.item()]], dtype=torch.long, device=device
        ),
        torch.tensor([[0, 1, EOS_token.item()]], dtype=torch.long, device=device),
    ]

    for idx, example in enumerate(examples):
        runner._predict(max_length=max_length, src=example)
