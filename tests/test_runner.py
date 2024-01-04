from pytorch_lightning.trainer import Trainer

from llm_non_identifiability.datamodule import GrammarDataModule
from llm_non_identifiability.runner import LightningGrammarModule

import torch


def test_fit(n_train, n_val, n_test):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule()
    dm = GrammarDataModule(n_train=n_train, n_val=n_val, n_test=n_test)
    trainer.fit(runner, datamodule=dm)


def test_predict(n_train, n_val, n_test):
    trainer = Trainer(fast_dev_run=True)
    runner = LightningGrammarModule()
    dm = GrammarDataModule(n_train=n_train, n_val=n_val, n_test=n_test)
    trainer.fit(runner, datamodule=dm)

    trainer.predict(runner, datamodule=dm)


def test_predict_inner(max_length):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runner = LightningGrammarModule()

    # Here we test some examples to observe how the model predicts
    examples = [
        torch.tensor([[2, 0, 0, 0, 0, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 0, 0, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device),
    ]

    for idx, example in enumerate(examples):
        runner._predict(max_length=max_length, src=example)
