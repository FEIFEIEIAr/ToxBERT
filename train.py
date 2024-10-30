import pandas as pd
import args
import os

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR,  CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryConfusionMatrix, BinaryROC

from data_module import SmilesDataModule
from tokenizer import Tokenizer
from util import *
from constant import MAX_LENGTH, BOS_ID, EOS_ID, PAD_ID, IGN_ID


class ELECTRAModel(nn.Module):
    def __init__(self, generator, discriminator, hf_tokenizer, sampling='fp32_gumbel'):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.hf_tokenizer = hf_tokenizer
        self.sampling = sampling

    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        device, dtype = a_tensor.device, a_tensor.dtype
        if self.sampling == 'fp32_gumbel':
            dtype = torch.float32
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(
            0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

    def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        sentA_lenths (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        attention_mask, _ = self._get_pad_mask_and_token_type(masked_inputs, sentA_lenths)
        gen_logits = self.generator(masked_inputs, attention_mask)[0]  # (B, L, vocab size)
        # reduce size to save space and speed
        # ( #mlm_positions, vocab_size)
        mlm_gen_logits = gen_logits[is_mlm_applied, :]

        with torch.no_grad():
            # sampling
            pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )
            # produce inputs for discriminator
            generated = masked_inputs.clone()  # (B,L)
            generated[is_mlm_applied] = pred_toks  # (B,L)
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone()  # (B,L)
            is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied])  # (B,L)

        disc_logits = self.discriminator(
            generated, attention_mask)[0]  # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.hf_tokenizer.pad_token_id
        seq_len = input_ids.shape[1]
        token_type_ids = torch.tensor([([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],
                                      device=input_ids.device)
        return attention_mask, token_type_ids

    def sample(self, logits):
        "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
        if self.sampling == 'fp32_gumbel':
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits.float() + gumbel).argmax(dim=-1)
        elif self.sampling == 'fp16_gumbel':  # 5.06 ms
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits + gumbel).argmax(dim=-1)
        elif self.sampling == 'multinomial':  # 2.X ms
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight=[1, 1], reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - self.weight[1] * target * torch.log(input) - (
            1 - target) * self.weight[0] * torch.log(1 - input)
        if self.reduction == 'mean':
            return torch.mean(bce)
        elif self.reduction == 'sum':
            return torch.sum(bce)


class ELECTRALoss():
    def __init__(self, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
        self.loss_weights = loss_weights
        # self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
        self.gen_loss_fc = nn.CrossEntropyLoss(ignore_index=IGN_ID)
        # self.disc_loss_fc = BCELoss_class_weighted([1, 1])
        self.disc_loss_fc = BCELoss_class_weighted(reduction='mean')
        self.disc_label_smooth = disc_label_smooth

    def __call__(self, pred, targ_ids):
        mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred

        gen_loss = self.gen_loss_fc(mlm_gen_logits.float().view(-1, mlm_gen_logits.size(-1)), targ_ids.reshape(-1))

        # select all non-pad tokens
        disc_logits = disc_logits.masked_select(non_pad)  # -> 1d tensor
        is_replaced = is_replaced.masked_select(non_pad)  # -> 1d tensor

        disc_logits = F.sigmoid(disc_logits)

        if self.disc_label_smooth:
            is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)

        disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())

        loss = gen_loss * self.loss_weights[0] + \
            disc_loss * self.loss_weights[1]
        return loss, gen_loss, disc_loss


class LightningModule(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()
        # Init hyperparameters
        self.hparams.update(vars(config))

        # Word embeddings layer
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # model
        gen_config = ElectraConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_size=self.hparams.embedding_dim,
            hidden_size=256,
            num_hidden_layers=self.hparams.n_layers//4,
            num_attention_heads=self.hparams.n_heads,
            intermediate_size=128,
            hidden_act="gelu",  # gelu_new
            hidden_dropout_prob=self.hparams.dropout,
            attention_probs_dropout_prob=self.hparams.dropout,
            max_position_embeddings=512,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,)

        disc_config = ElectraConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_size=self.hparams.embedding_dim,
            hidden_size=256,
            num_hidden_layers=self.hparams.n_layers,
            num_attention_heads=self.hparams.n_heads,
            intermediate_size=128,
            hidden_act="gelu",  # gelu_new
            hidden_dropout_prob=self.hparams.dropout,
            attention_probs_dropout_prob=self.hparams.dropout,
            max_position_embeddings=512,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,)

        generator = ElectraForMaskedLM(gen_config)
        discriminator = ElectraForPreTraining(disc_config)
        discriminator.electra.embeddings = generator.electra.embeddings

        generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

        # ELECTRA training loop
        self.electra_model = ELECTRAModel(generator, discriminator, tokenizer)
        self.electra_loss_func = ELECTRALoss(
            gen_label_smooth=self.hparams.gen_smooth_label, disc_label_smooth=self.hparams.disc_smooth_label)

        # self.init_weights(self.electra_model)
        
        # Store outputs for on_the_epoch_end hooks
        self.training_outputs = []
        self.validation_outputs = [[], []]

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, X, X_mask, Y, token_type_ids=None, masked_indices=None):
        """Forward for ELECTRA model.

        Args:
            X (Tensor): Masked input ids, shape (batch_size, seq_len).
            X_mask (Tensor): Length mask for inputs, shape (batch_size, seq_len).
            Y (Tensor): Original labels, shape (batch_size, seq_len).
            token_type_ids (_type_, optional): _description_. Defaults to None.
            masked_indices (_type_, optional): _description_. Defaults to None.

        Returns:
            mlm_gen_logits
            generated
            disc_logits
            is_replaced (Tensor): Bool. True for replaced tokens, False for original tokens.
            X_mask
            masked_indices
        """
        X_mask = X_mask.bool()
        gen_logits = self.electra_model.generator(X, X_mask, token_type_ids)[0]  # (B, L, vocab size)
        # reduce size to save space and speed
        # mlm_gen_logits = gen_logits[masked_indices, :] # ( #mlm_positions, vocab_size)
        mlm_gen_logits = gen_logits

        with torch.no_grad():
            # sampling
            pred_toks = self.electra_model.sample(
                mlm_gen_logits[masked_indices, :])  # ( #mlm_positions, )
            # produce inputs for discriminator
            generated = X.clone()  # (B,L)
            generated[masked_indices] = pred_toks  # (B,L)
            # produce labels for discriminator
            is_replaced = masked_indices.clone()  # (B,L)
            is_replaced[masked_indices] = (pred_toks != Y[masked_indices])  # (B,L)

        disc_logits = self.electra_model.discriminator(generated, X_mask, token_type_ids)[0]  # (B, L)
        return mlm_gen_logits, generated, disc_logits, is_replaced, X_mask, masked_indices

    def calc_metrics(self, disc_logits, label, X_mask, threshold=0.6):
        disc_logits = F.sigmoid(disc_logits)

        device = disc_logits.device
        label = label.clone().detach().to(disc_logits.device).float()
        label = label.masked_fill(~X_mask.bool(), IGN_ID)

        auroc_metric = BinaryAUROC(ignore_index=IGN_ID).to(self.device)
        confmat_metric = BinaryConfusionMatrix(threshold=threshold, ignore_index=IGN_ID).to(self.device)
        confmat = confmat_metric(disc_logits, label)

        # 计算 TP, TN, FP, FN
        TN = confmat[0, 0].item()
        FP = confmat[0, 1].item()
        FN = confmat[1, 0].item()
        TP = confmat[1, 1].item()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        auroc = auroc_metric(disc_logits, label)
        mcc = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0 else 0

        self.log_dict({"acc": accuracy,
                       "AUROC": auroc,
                       "mcc": mcc
                       }, prog_bar=True, logger=True, add_dataloader_idx=False)

    def training_step(self, batch, batch_idx):
        """
        Implement the training step for pytorch lightning.

        Args:
            batch (Tuple): batch was compose of (X, X_mask, Y, masked_indices)
                X (Tensor): Inputs ids, shape (batch_size, seq_len).
                length_mask (Tensor): Attention mask for inputs, also call x length mask, shape (batch_size, seq_len).
                Y (Tensor): Labels ids, shape (batch_size, seq_len).
                masked_indices (Tensor): Indices of mask indicate tokens that will be calc loss.
                token_type_ids (Tensor): Token type ids for inputs, shape (batch_size, seq_len).
                label (Tensor): Labels for drugs, shape (batch_size, ).
                original_input (Tensor): Original unmasked inputs for drugs, shape (batch_size, seq_len).
                special_tokens_mask (Tensor): special_tokens_mask.
        """
        X, length_mask, Y, masked_indices, token_type_ids, _, _, _ = batch

        # Get normal loss
        pred = self.forward(X, length_mask, Y, token_type_ids=token_type_ids, masked_indices=masked_indices)
        mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
        self.calc_metrics(disc_logits, is_replaced, length_mask)
        loss, gen_loss, disc_loss = self.electra_loss_func(pred, Y)

        self.log_dict({"loss": loss,
                       "gen_loss": gen_loss,
                       "disc_loss": disc_loss}, prog_bar=True, logger=True)
        return loss

    def clip(self, disc_logits, threshold=0.5):
        if not (0 < threshold < 1):
            raise ValueError("threshold must be in [0, 1]")
        target = disc_logits <= threshold
        disc_logits = disc_logits.masked_fill(target.bool(), 0)
        disc_logits = disc_logits.masked_fill(~target.bool(), 1)
        return disc_logits

    def calc_valid_metric(self, logits, label, threshold):        
        auroc_metric = BinaryAUROC().to(self.device)
        auprc_metric = BinaryAveragePrecision().to(self.device)
        confmat_metric = BinaryConfusionMatrix(threshold=threshold).to(self.device)
        confmat = confmat_metric(logits, label)

        # 计算 TP, TN, FP, FN
        TN = confmat[0, 0].item()
        FP = confmat[0, 1].item()
        FN = confmat[1, 0].item()
        TP = confmat[1, 1].item()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        auroc = auroc_metric(logits, label)
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        mcc = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        bacc = (recall + specificity) / 2
        auprc = auprc_metric(logits, label)
        return accuracy, auroc, F1, precision, recall, mcc, specificity, bacc, auprc
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """        
        Args:
            batch (Tuple): batch was compose of (X, X_mask, Y, masked_indices)
                X (Tensor): Inputs ids, shape (batch_size, seq_len).
                length_mask (Tensor): Attention mask for inputs, also call x length mask, shape (batch_size, seq_len).
                Y (Tensor): Labels ids, shape (batch_size, seq_len).
                masked_indices (Tensor): Indices of mask indicate tokens that will be calc loss.
                token_type_ids (Tensor): Token type ids for inputs, shape (batch_size, seq_len).
                label (Tensor): Labels for drugs, shape (batch_size, ).
                original_input (Tensor): Original unmasked inputs for drugs, shape (batch_size, seq_len).
                special_tokens_mask (Tensor): special_tokens_mask.
        """
        _, length_mask, _, _, token_type_id, label, origin_input, _ = batch
        dataset = self.hparams.dataset_names[dataloader_idx]
        disc_logits = self.electra_model.discriminator(
            origin_input, length_mask, token_type_id)[0]  # (B, L)
        # disc_logits = self.electra_model.discriminator(
        #     origin_input, length_mask, )[0]  # (B, L)
        device = disc_logits.device
        label = torch.tensor(label).to(self.device).long()

        disc_logits = F.sigmoid(disc_logits)
        
        if self.hparams.threshold > 0:
            disc_logits = self.clip(disc_logits, threshold=self.hparams.threshold)
            
        disc_logits = disc_logits * length_mask
        average_disc_logits = disc_logits.sum(dim=-1) / length_mask.sum(dim=-1)

        # Log data per validation epoch for personal analysis
        self.validation_outputs[dataloader_idx].append({
            "disc_logits": average_disc_logits,
            "label": label,
            "dataset_idx": dataloader_idx,
            "count": disc_logits.sum(dim=-1),
            "seq_length": length_mask.sum(dim=-1),
        })
        return

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        for dataloader_idx, batch_outputs in enumerate(outputs):
            dataset = self.hparams.dataset_names[dataloader_idx]

            self.smiles_path = os.path.join(self.logger.log_dir, 'smiles')
            os.mkdir(self.smiles_path) if not os.path.exists(self.smiles_path) else None
            disc_logits = [item for x in batch_outputs for item in x["disc_logits"].tolist()]
            labels = [item for x in batch_outputs for item in x["label"].tolist()]
            
            # J index
            roc = BinaryROC(thresholds=None)
            disc_logits = torch.tensor(disc_logits).to(self.device)
            labels = torch.tensor(labels).to(self.device)
            fpr, tpr, thresholds = roc(disc_logits, labels)
            youdens_j = tpr - fpr
            best_threshold_idx = torch.argmax(youdens_j).item()
            best_threshold = thresholds[best_threshold_idx].item()
            
            self.log(f"{dataset}_best_threshold", best_threshold, logger=True, add_dataloader_idx=False)
            accuracy, auroc, F1, precision, recall, mcc, specificity, bacc, auprc = self.calc_valid_metric(disc_logits, labels, threshold=best_threshold)
            self.log("{0}_acc".format(dataset), accuracy,prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_AUROC".format(dataset), auroc, prog_bar=True, logger=True, add_dataloader_idx=False)
            self.log("{0}_F1".format(dataset), F1, prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_precision".format(dataset), precision,prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_recall".format(dataset), recall,prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_mcc".format(dataset), mcc,prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_specificity".format(dataset), specificity,prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_bacc".format(dataset), bacc,prog_bar=False, logger=True, add_dataloader_idx=False)
            self.log("{0}_auprc".format(dataset), auprc,prog_bar=False, logger=True, add_dataloader_idx=False)

            self.validation_outputs[dataloader_idx].clear()

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optimizer = optim.AdamW(self.electra_model.parameters(),
                                    lr=self.hparams.lr,
                                    betas=(self.hparams.beta1, self.hparams.beta2),
                                    weight_decay=self.hparams.weight_decay,
                                    eps=1e-6,
                                    amsgrad=True)
        elif self.hparams.optimizer == "RMSprop":
            optimizer = optim.RMSprop(self.electra_model.parameters(),
                                      lr=self.hparams.lr,
                                      alpha=0.9,
                                      eps=1e-6,
                                      weight_decay=self.hparams.weight_decay,
                                      momentum=0.9,
                                      centered=False)
        elif self.hparams.optimizer == "SGD":
            optimizer = optim.SGD(self.electra_model.parameters(),
                                  lr=self.hparams.lr,
                                  weight_decay=self.hparams.weight_decay,
                                  momentum=0.9)

        if self.hparams.scheduler == "MultiStepLR":
            scheduler = MultiStepLR(
                optimizer, milestones=self.hparams.decay_steps, gamma=self.hparams.decay_gamma)
        elif self.hparams.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs)

        return [optimizer], [scheduler,]


def main():
    margs = args.parse_args()
    margs.dataset_names = ["valid", "test"]
    pl.seed_everything(margs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer_path = 'bert_vocab_qt.txt'
    tokenizer = Tokenizer(tokenizer_path)

    margs.vocab_size = tokenizer.vocab_size
    if margs.checkpoint_every > margs.max_epochs:
        margs.checkpoint_every = margs.max_epochs

    margs.gen_smooth_label = False
    margs.disc_smooth_label = False
    margs.sampling = "multinomial"
    margs.electra_mask_style = True

    if margs.checkpoint_path == '':
        print("# training from scratch")
        model = LightningModule(margs, tokenizer=tokenizer)
    else:
        if os.path.isdir(margs.checkpoint_path):
            margs.checkpoint_path = os.path.join(
                margs.checkpoint_path, os.listdir(margs.checkpoint_path)[0])
        print(
            "# loaded pre-trained model from {0}".format(margs.checkpoint_path))
        model = LightningModule(margs, tokenizer=tokenizer).load_from_checkpoint(
            margs.checkpoint_path, strict=False, config=margs, tokenizer=tokenizer, vocab_size=len(tokenizer.vocab))
    if margs.drop_norm > 0:
        assert margs.drop_norm <= 1
        remove_norm_layers(model, margs.norm)

    print(margs)
    logger = TensorBoardLogger(margs.checkpoints_folder,
                               name=margs.dataset_name,
                               default_hp_metric=False,
                               version="{}lr_{}weight_decay_{}embedding_dim_{}n_layer_{}".format(margs.dataset_name, margs.lr, margs.weight_decay, margs.embedding_dim, margs.n_layers))
    
    data_module = SmilesDataModule(margs, tokenizer)
    
    trainer = pl.Trainer(max_epochs=margs.max_epochs,
                         accelerator="auto",
                         devices="auto" if torch.cuda.is_available() else None,
                         log_every_n_steps=5,
                         check_val_every_n_epoch=margs.valid_every,
                         logger=logger,
                         accumulate_grad_batches=margs.accumulate_grad_batches,
                         #  callbacks=[checkpoint_callback],
                         #  sync_batchnorm=True,
                         enable_progress_bar=True,
                         precision="16-mixed",
                         )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
