import math
import subprocess
from os.path import dirname
from typing import Optional, Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers.optimization import get_inverse_sqrt_schedule

from llm_non_identifiability.data import (
    check_same_number_as_bs,
    check_as_before_bs,
    SOS_token,
    EOS_token,
    PAD_token,
    check_sequence_finished,
    generate_test_prompts,
    grammar_rules,
    prompt_grammar_rules,
    GrammarMetrics,
    pad,
)
from llm_non_identifiability.model import (
    TransformerDecoder,
    create_pad_mask,
    get_tgt_mask,
)


class LightningGrammarModule(pl.LightningModule):
    """
    LightningModule for training a Transformer on sequence data coming from a PCFG grammar.
    """

    def __init__(
        self,
        num_tokens: int = 5,
        dim_model: int = 8,
        dim_feedforward: int = 256,
        num_heads: int = 4,
        num_decoder_layers: int = 2,
        max_pred_length: int = 64,
        test_prompt_length: int = 6,
        dropout_p: float = 0.1,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offline: bool = False,
        next_token_pick_mode: str = "max",
        layer_norm_eps: float = 2e-4,
        grammar: str = "aNbN",
        max_data_length: int = 256,
        batch_size: int = 64,
        relu_rescale: float = 1.0,
        adversarial_training: bool = False,
        num_warmup_steps: int = 1000,
        extrapolation_training: bool = False,
        optimizer: str = "adamw",
    ):
        """
        :param optimizer:
        :param extrapolation_training:
        :param num_warmup_steps:
        :param relu_rescale:
        :param adversarial_training:
        :param batch_size:
        :param max_data_length:
        :param grammar:
        :param layer_norm_eps:
        :param next_token_pick_mode:
        :param dim_feedforward:
        :param test_prompt_length:
        :param max_pred_length:
        :param offline:
        :param lr: learning rate
        :param device:
        """
        super().__init__()
        self.save_hyperparameters()

        if (
            self.hparams.extrapolation_training is True
            and self.hparams.adversarial_training is True
        ):
            raise ValueError(
                "Cannot train with both extrapolation and adversarial training"
            )

        self.hparams["loss_fn"] = nn.CrossEntropyLoss()
        self.model = TransformerDecoder(
            num_tokens=self.hparams.num_tokens,
            dim_model=self.hparams.dim_model,
            num_heads=self.hparams.num_heads,
            num_decoder_layers=self.hparams.num_decoder_layers,
            dropout_p=self.hparams.dropout_p,
            dim_feedforward=self.hparams.dim_feedforward,
            layer_norm_eps=self.hparams.layer_norm_eps,
            relu_rescale=self.hparams.relu_rescale,
        )

        self.grammar_rules = grammar_rules(self.hparams.grammar)
        self.prompt_grammar_rules = prompt_grammar_rules(self.hparams.grammar)
        self._setup_test_prompts()

    @property
    def data_entropy(self):
        return math.log(n := (self.hparams.max_data_length // 2), math.e) / n

    def configure_optimizers(self):
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
            return {
                "optimizer": optimizer,
                "lr_scheduler": get_inverse_sqrt_schedule(
                    optimizer=optimizer, num_warmup_steps=self.hparams.num_warmup_steps
                ),
            }
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

    def _setup_test_prompts(self) -> None:
        test_prompts = generate_test_prompts(length=self.hparams.test_prompt_length).to(
            self.hparams.device
        )

        rules_met = [self.prompt_grammar_rules(t) for t in test_prompts]

        self.test_prompts_in_distribution = test_prompts[rules_met]
        self.test_prompts_out_of_distribution = test_prompts[[not r for r in rules_met]]

        self.hparams.test_prompts_in_distribution_len = len(
            self.test_prompts_in_distribution
        )
        self.hparams.test_prompts_ood_len = len(self.test_prompts_out_of_distribution)

        assert (
            len(test_prompts)
            == self.hparams.test_prompts_in_distribution_len
            + self.hparams.test_prompts_ood_len
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            # log entropy of the test prompts = entropy of the distribution of the prompt lengths
            # log as a summary item
            self.logger.experiment.summary["data_entropy"] = self.data_entropy

        self.__setup_adversarial_prompts()
        self.__setup_extrapolation_prompts()

    def __setup_adversarial_prompts(self) -> None:
        """
        Setup the prompts for adversarial training from the OOD test prompts
        """

        if self.hparams.adversarial_training is True:
            prompts = []

            for idx, prompt in enumerate(self.test_prompts_out_of_distribution):
                num_as = torch.sum(prompt == 0)
                num_bs = torch.sum(prompt == 1)

                if num_as >= num_bs:
                    prompt = self._extend_prompt(
                        prompt, num_as - num_bs + 1, value=torch.ones
                    )
                else:
                    prompt = self._extend_prompt(
                        prompt, num_bs - num_as + 1, value=torch.zeros
                    )

                prompts.append(prompt.cpu().numpy())

            self.adversarial_prompts = (
                torch.from_numpy(pad(prompts)).long().to(self.hparams.device)
            )

    def __setup_extrapolation_prompts(self) -> None:
        """
        Setup the prompts for extrapolation training from the OOD test prompts
        """

        if self.hparams.extrapolation_training is True:
            prompts = []

            for idx, prompt in enumerate(self.test_prompts_out_of_distribution):
                num_as = torch.sum(prompt == 0)
                num_bs = torch.sum(prompt == 1)

                if num_as >= num_bs:
                    prompt = self._extend_prompt(
                        prompt, num_as - num_bs, value=torch.ones
                    )
                else:
                    prompt = self._extend_prompt(
                        prompt, num_bs - num_as, value=torch.zeros
                    )

                assert check_same_number_as_bs(prompt) == True

                prompts.append(prompt.cpu().numpy())

            self.extrapolation_prompts = (
                torch.from_numpy(pad(prompts)).long().to(self.hparams.device)
            )

    def _extend_prompt(self, prompt, length, value=torch.ones):
        prompt = torch.cat(
            (
                prompt,
                value(
                    (length,),
                    dtype=torch.long,
                    device=self.hparams.device,
                ),
                torch.ones(
                    (1,),
                    dtype=torch.long,
                    device=self.hparams.device,
                )
                * EOS_token.item(),
            ),
            dim=0,
        )
        return prompt

    def training_step(self, batch, batch_idx):
        panel_name = "Train"
        _, _, _, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        if self.hparams.adversarial_training is True:
            _, _, _, loss_adversarial = self._forward(
                self.adversarial_prompts, completion_loss=True
            )
            self.log(f"{panel_name}/loss_adversarial", loss_adversarial)

            with torch.no_grad():
                _, _, _, loss_adversarial_full = self._forward(
                    self.adversarial_prompts, completion_loss=False
                )
                self.log(
                    f"{panel_name}/loss_adversarial_prompt",
                    loss_adversarial_full - loss_adversarial,
                )

            loss += loss_adversarial

        if self.hparams.extrapolation_training is True:
            _, _, _, loss_extrapolation = self._forward(
                self.extrapolation_prompts, completion_loss=True
            )
            self.log(f"{panel_name}/loss_extrapolation", loss_extrapolation)

            with torch.no_grad():
                _, _, _, loss_extrapolation_full = self._forward(
                    self.extrapolation_prompts, completion_loss=False
                )
                self.log(
                    f"{panel_name}/loss_extrapolation_prompt",
                    loss_extrapolation_full - loss_extrapolation,
                )

            loss += loss_extrapolation

        return loss

    def validation_step(self, batch, batch_idx):
        panel_name = "Val"
        X, X_expected, pred, loss = self._forward(batch)
        self.log(f"{panel_name}/loss", loss)

        self.log(f"{panel_name}/kl", loss - self.data_entropy)

        # pick most likely token and calculate and log accuracy
        pred_tokens = self._pick_next_tokens(pred)
        accuracy = torch.sum(pred_tokens == X_expected) / X_expected.numel()
        self.log(f"{panel_name}/accuracy", accuracy)

        (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        ) = self.eval_prompt_prediction()

        self._log_dict(name=f"{panel_name}/ID", dictionary=metrics.to_dict())
        self._log_dict(
            name=f"{panel_name}/ID/finished", dictionary=metrics_finished.to_dict()
        )

        self._log_dict(name=f"{panel_name}/OOD", dictionary=ood_metrics.to_dict())
        self._log_dict(
            name=f"{panel_name}/OOD/finished", dictionary=ood_metrics_finished.to_dict()
        )
        self._log_dict(name=f"{panel_name}/SOS", dictionary=sos_metrics.to_dict())
        self._log_dict(
            name=f"{panel_name}/SOS/finished", dictionary=sos_metrics_finished.to_dict()
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            logger: pl.loggers.wandb.WandbLogger = self.logger

            # log the prompts
            prompts2str = lambda data: [
                ["".join([str(t) for t in p])] for p in data.cpu().numpy().tolist()
            ]
            # convert the prompt tensors to strings
            prompts_str = prompts2str(prompts)
            ood_prompts_str = prompts2str(ood_prompts)
            sos_prompts_str = prompts2str(sos_prompts)

            prompts_finished_str = prompts2str(prompts_finished)
            ood_prompts_finished_str = prompts2str(ood_prompts_finished)
            sos_prompts_finished_str = prompts2str(sos_prompts_finished)

            columns = ["completion"]

            # data should be a list of lists
            logger.log_text(
                key="id_prompt_completions", columns=columns, data=prompts_str
            )
            logger.log_text(
                key="ood_prompt_completions", columns=columns, data=ood_prompts_str
            )
            logger.log_text(
                key="sos_prompt_completions", columns=columns, data=sos_prompts_str
            )

            logger.log_text(
                key="id_prompt_completions_finished",
                columns=columns,
                data=prompts_finished_str,
            )
            logger.log_text(
                key="ood_prompt_completions_finished",
                columns=columns,
                data=ood_prompts_finished_str,
            )
            logger.log_text(
                key="sos_prompt_completions_finished",
                columns=columns,
                data=sos_prompts_finished_str,
            )

    def _log_dict(self, name, dictionary):
        for key, value in dictionary.items():
            self.log(f"{name}/{key}", value)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        ) = self.eval_prompt_prediction()

        checkpoint["prompts"] = prompts.cpu().numpy()
        checkpoint["prompts_finished"] = prompts_finished.cpu().numpy()

        checkpoint["ood_prompts"] = ood_prompts.cpu().numpy()
        checkpoint["ood_prompts_finished"] = ood_prompts_finished.cpu().numpy()

        checkpoint["sos_prompts"] = sos_prompts.cpu().numpy()
        checkpoint["sos_prompts_finished"] = sos_prompts_finished.cpu().numpy()

    @property
    def test_prompts_src(self):
        ds = self.trainer.datamodule.test_dataset.data.view(-1)
        return ds[ds != PAD_token.item()].long().to(self.hparams.device)

    def eval_prompt_prediction(self, max_length: Optional[int] = None):
        if max_length is None:
            max_length = self.hparams.max_pred_length

        (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_in_distribution, max_length
        )

        (
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
        ) = self._calc_prompt_pred_metrics(
            self.test_prompts_out_of_distribution, max_length
        )

        # prompt prediction for a batch of SOS tokens
        sos_prompts = (
            torch.ones(
                (self.hparams.batch_size, 1),
                dtype=torch.long,
                device=self.hparams.device,
            )
            * SOS_token.item()
        )
        (
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        ) = self._calc_prompt_pred_metrics(sos_prompts, max_length)

        return (
            prompts,
            metrics,
            prompts_finished,
            metrics_finished,
            ood_prompts,
            ood_metrics,
            ood_prompts_finished,
            ood_metrics_finished,
            sos_prompts,
            sos_metrics,
            sos_prompts_finished,
            sos_metrics_finished,
        )

    def _calc_prompt_pred_metrics(self, prompts, max_length):
        prompt_pred = self._predict(max_length=max_length, prompt=prompts)

        metrics, finished = self._calc_grammar_metrics(prompt_pred)

        if sum(finished) == 0:
            prompt_pred_finished = torch.tensor(
                [], dtype=torch.long, device=self.hparams.device
            )
            metrics_finished = GrammarMetrics()
        else:
            # filter out finsihed prompts only
            prompt_pred_finished = torch.stack(
                [p for p, f in zip(prompt_pred, finished) if f == True]
            )

            metrics_finished, _ = self._calc_grammar_metrics(prompt_pred_finished)

        return (
            prompt_pred,
            metrics,
            prompt_pred_finished,
            metrics_finished,
        )

    def _calc_grammar_metrics(self, prompt_pred, eps: float = 1e-8):
        as_before_bs = [check_as_before_bs(p) for p in prompt_pred]
        same_number_as_bs = [check_same_number_as_bs(p) for p in prompt_pred]
        grammatical = [self.grammar_rules(p) for p in prompt_pred]
        finished = [check_sequence_finished(p) for p in prompt_pred]
        as_before_bs_completion = [
            check_as_before_bs(
                p[self.hparams.test_prompt_length + 1 :]
            )  # +1 is for the SOS token
            for p in prompt_pred
        ]

        metrics = GrammarMetrics(
            as_before_bs_accuracy=sum(as_before_bs) / (len(as_before_bs) + eps),
            same_number_as_bs_accuracy=sum(same_number_as_bs)
            / (len(same_number_as_bs) + eps),
            finished_accuracy=sum(finished) / (len(finished) + eps),
            grammatical_accuracy=sum(grammatical) / (len(grammatical) + eps),
            as_before_bs_completion_accuracy=sum(as_before_bs_completion)
            / (len(as_before_bs_completion) + eps),
        )

        return metrics, finished

    def _forward(self, X, completion_loss=False):
        """
        Forward pass for calculating the model predictions and the loss.
        :param completion_loss: calculate loss only on prompt completion
        :param X:
        :return:
        """

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        X_input = X[:, :-1]
        X_expected = X[:, 1:]

        # Get mask to mask out the next words
        causal_mask = get_tgt_mask(X_input.size(1), device=self.hparams.device)

        # Standard training except we pass in X_input and causal_mask
        pred = self.model(
            src=X_input,
            mask=causal_mask,
            src_key_padding_mask=create_pad_mask(X_input),
        )

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)

        if completion_loss is False:
            loss = self.hparams.loss_fn(pred, X_expected)
        else:
            loss = self.hparams.loss_fn(
                pred[
                    :, :, self.hparams.test_prompt_length + 1 :
                ],  # +1 is for the SOS token
                X_expected[:, self.hparams.test_prompt_length + 1 :],
            )

        return X, X_expected, pred, loss

    def predict_step(  # type: ignore
        self,
        batch,
        batch_idx=None,
        prompt: Optional[torch.Tensor] = None,
        max_length: int = 32,
    ):
        """
        Overriding the default method of the LightningModule for being called with Trainer.predict().
        :param batch:
        :param batch_idx:
        :param prompt: optional prompt to start the prediction
        :param max_length: maximum sequence length for the prediction
        :return:
        """

        return self._predict(max_length, prompt)

    def _predict(
        self, max_length: int = 32, prompt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inner method for predicting a sequence.
        :param max_length: maximum sequence length for the prediction
        :param prompt: optional prompt to start the prediction
        :return:
        """

        if prompt is None:
            prompt = torch.tensor(
                [[0, 0, 0, 1]],
                dtype=torch.long,
                device=self.hparams.device,  # type: ignore
            )

        finished = torch.BoolTensor([False] * prompt.size(0)).to(self.hparams.device)

        for _ in range(max_length):
            # Get mask to mask out the next words
            tgt_mask = get_tgt_mask(size=(prompt.size(1)), device=self.hparams.device)

            # forward pass
            pred = self.model(
                src=prompt,
                mask=tgt_mask,
                src_key_padding_mask=create_pad_mask(prompt),
            )

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)

            # pick the prediction for the last token only
            next_items = self._pick_next_tokens(pred)[:, -1].view(-1, 1)

            # Concatenate previous input with predicted best word
            prompt = torch.cat((prompt, next_items), dim=1)

            # save if model predicts end of sentence
            finished.logical_or_(next_items.view(-1) == EOS_token.item())
            # Stop if model predicts end of sentence
            if torch.all(finished) is True:
                break
        return prompt.long().to(self.hparams.device)

    def _pick_next_tokens(self, pred: torch.Tensor) -> torch.Tensor:
        if self.hparams.next_token_pick_mode == "max":
            _, next_items = torch.max(pred, dim=1)
        elif self.hparams.next_token_pick_mode == "sample":
            if len(pred.shape) > 2 and len(pred.squeeze().shape) != 2:
                next_items = torch.cat(
                    [
                        torch.multinomial(
                            torch.softmax(p.squeeze(), dim=1).T, num_samples=1
                        ).T
                        for p in pred
                    ]
                )
            else:
                next_items = torch.multinomial(
                    torch.softmax(pred.squeeze(), dim=1).T, num_samples=1
                ).T
        else:
            raise ValueError(
                f"Unknown next_token_pick_mode: {self.hparams.next_token_pick_mode}, should be 'max' or 'sample'"
            )
        return next_items.to(self.hparams.device)

    def on_fit_end(self) -> None:
        self._sync_wandb()

    def _sync_wandb(self):
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            logger: pl.loggers.wandb.WandbLogger = self.logger  # type: ignore
            if self.hparams.offline is True:
                # Syncing W&B at the end
                # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
                sync_dir = dirname(logger.experiment.dir)
                # 2. mark run complete
                wandb.finish()  # type: ignore
                # 3. call the sync command for the run directory
                subprocess.check_call(["wandb", "sync", sync_dir])
