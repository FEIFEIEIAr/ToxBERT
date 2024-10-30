import os
import random

import torch
import torch.utils.data
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from pandarallel import pandarallel

import args
from util import *
from tokenizer import Tokenizer
from constant import MAX_LENGTH, BOS_ID, EOS_ID, PAD_ID, MASK_ID, IGN_ID
from mask import TokenMask, SpanMask, NERSpanMask

pandarallel.initialize()


def get_dataset(data_root, filename,  dataset_len, measure_name='label', randomize_smiles=True):
    df = pd.read_csv(os.path.join(data_root, filename), usecols=['canonical_smiles', measure_name])
    print("Length of dataset:", len(df))
    
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = SmilesDataset(df=df, measure_name=measure_name)
    return dataset

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name='label'):
        # Read SMILES
        df_good = df.dropna(subset=['canonical_smiles'])
        df_good = df_good.dropna(subset=[measure_name])
        df_good = df_good.drop_duplicates(subset=['canonical_smiles'])
        df_good = df_good.reset_index(drop=True)
        self.measure_name = measure_name
        
        print("Drop invalid SMILES {}".format(len(df) - len(df_good)))
        self.df = df_good

    def __getitem__(self, index):
        smiles = self.df.loc[index, 'canonical_smiles']
        label = self.df.loc[index, self.measure_name]
        return smiles, label

    def __len__(self):
        return len(self.df)
    
class SmilesDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer):
        super(SmilesDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.mlm_probability = self.hparams.mask_prob
        self.char2id = {"<bos>":BOS_ID, "<eos>":EOS_ID, "<pad>":PAD_ID, "<mask>":MASK_ID}
        self.mask = self.get_mask()

    def get_split_dataset_filename(self, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = self.get_split_dataset_filename("train")
        valid_filename = self.get_split_dataset_filename("valid")
        test_filename = self.get_split_dataset_filename("test")

        train_ds = get_dataset(
            self.hparams.data_root,
            train_filename,
            None,
            measure_name=self.hparams.measure_name,
        )

        val_ds = get_dataset(
            self.hparams.data_root,
            valid_filename,
            None,
            measure_name=self.hparams.measure_name,
        )

        test_ds = get_dataset(
            self.hparams.data_root,
            test_filename,
            None,
            measure_name=self.hparams.measure_name,
        )

        self.train_ds = train_ds
        self.valid_ds = val_ds
        self.val_ds = [val_ds] + [test_ds]
        
    def get_mask(self):
        if self.hparams.mask == "token":
            return TokenMask(
                self.mlm_probability, 
                self.tokenizer)
        elif self.hparams.mask == "span":
            return SpanMask(
                self.mlm_probability, 
                self.tokenizer,
                self.hparams.lower,
                self.hparams.upper,
                self.hparams.replacement,
                self.hparams.geometric_p)
        # elif self.hparams.mask == "ner":
        #     return NERSpanMask(self.mlm_probability, self.tokenizer)
        else:
            raise ValueError("Mask {} not supported".format(self.hparams.mask))

    def collate(self, batch):
        smiles = [item[0] for item in batch]
        label = [item[1] for item in batch]
        smiles = self.tokenizer.batch_encode_plus([smile for smile in smiles], 
                                                   padding=True, 
                                                   add_special_tokens=True, 
                                                   max_length=MAX_LENGTH, 
                                                   truncation=True,
                                                   return_special_tokens_mask=True)
        
        inputs, labels, masked_indices = self.mask(torch.tensor(smiles['input_ids']),
                                                   torch.tensor(smiles['special_tokens_mask']))
        
        return (inputs, 
                torch.tensor(smiles['attention_mask']), 
                labels, 
                masked_indices, 
                torch.tensor(smiles['token_type_ids']), 
                label,
                torch.tensor(smiles['input_ids']),
                torch.tensor(smiles['special_tokens_mask']))
    
    def worker_init_fn(self, worker_id):
        seed = 0 + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                collate_fn=self.collate,
                worker_init_fn=self.worker_init_fn,
                drop_last=False,
            )
            for ds in self.val_ds
        ]
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
            worker_init_fn=self.worker_init_fn,
            # drop_last=True,
        )
         
class PropertyDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer):
        super(PropertyDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.mlm_probability = self.hparams.mask_prob
        self.char2id = {"<bos>":BOS_ID, "<eos>":EOS_ID, "<pad>":PAD_ID, "<mask>":MASK_ID}

    def get_split_dataset_filename(self, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = self.get_split_dataset_filename("train")
        valid_filename = self.get_split_dataset_filename("valid")
        test_filename = self.get_split_dataset_filename("test")

        train_ds = get_dataset(
            self.hparams.data_root,
            train_filename,
            self.hparams.train_dataset_length,
            measure_name=self.hparams.measure_name,
        )

        val_ds = get_dataset(
            self.hparams.data_root,
            valid_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
        )

        test_ds = get_dataset(
            self.hparams.data_root,
            test_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
        )

        self.train_ds = train_ds
        self.valid_ds = val_ds
        self.val_ds = [val_ds] + [test_ds]

    def collate(self, batch):
        smiles = [item[0] for item in batch]
        label = [item[1] for item in batch]
        smiles = self.tokenizer.batch_encode_plus([smile for smile in smiles], 
                                                   padding=True, 
                                                   add_special_tokens=True, 
                                                   max_length=MAX_LENGTH, 
                                                   truncation=True,
                                                   return_special_tokens_mask=True
                                                   )
        return (torch.tensor(smiles['input_ids']), 
                torch.tensor(smiles['attention_mask']), 
                torch.tensor(smiles['token_type_ids']), 
                torch.tensor(label))
        
    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                collate_fn=self.collate,
                worker_init_fn=self.worker_init_fn,
                drop_last=False,
            )
            for ds in self.val_ds
        ]
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
            worker_init_fn=self.worker_init_fn,
            # drop_last=True,
        )

if __name__ == "__main__":
    tokenizer = Tokenizer("bert_vocab.txt")
    # data = SmilesDataModule(0, tokenizer=tokenizer)
    smiles = [
    "C#CC1(CCCCC1)OC(=O)N", 
    "CC1=C(SC(=N1)C2=CC(=C(C=C2)OCC(C)C)C#N)C(=O)O",
    "C1CN(CCC1NC(=O)C2=CC=CC=C2C3=CC=C(C=C3)C(F)(F)F)CCCCC4(C5=CC=CC=C5C6=CC=CC=C64)C(=O)NCC(F)(F)F"]
    
    tokens = tokenizer.batch_encode_plus([smile for smile in smiles], 
                                         padding=True, 
                                         add_special_tokens=True,
                                         return_special_tokens_mask=True)
    
    inputs_ids = torch.tensor(tokens['input_ids'])
    attention_mask = torch.tensor(tokens["attention_mask"])
    inputs_ids = (torch.ones(inputs_ids.size())/2).masked_fill(~attention_mask.bool(), value=0)
    print(inputs_ids.sum(dim=1) / attention_mask.sum(dim=1))
    
    print(tokenizer.encode("<bos>C#CC1(CCCCC1)OC(=O)N"))
    # print(labels)
    # print('\n\n\n', inputs[0])
    # mask = 3
    # mlm_probability = 0.15
    # inputs = torch.tensor(tokens['input_ids'])
    # attention_mask = torch.tensor(tokens['attention_mask'])
    # special_tokens_mask = torch.tensor(tokens['special_tokens_mask'])
    
    # probability_matrix = torch.full(inputs.size(), mlm_probability)
    # print(probability_matrix)
    # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    # masked_indices = torch.bernoulli(probability_matrix).bool()
    # print(masked_indices)
    # random_words = torch.randint(4, tokenizer.vocab_size, inputs.size(), dtype=torch.long)
    # print(random_words)
    # print(tokenizer.vocab_size)