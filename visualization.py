import pandas as pd

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fastai.text.all import *
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
import yaml

import args
from tokenizer import Tokenizer
from util import *
from constant import MAX_LENGTH, BOS_ID, EOS_ID, PAD_ID, IGN_ID

class ELECTRAModel(nn.Module):
    def __init__(self, generator, discriminator, hf_tokenizer, sampling='fp32_gumbel'):
        super().__init__()
        self.generator, self.discriminator = generator,discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
        self.hf_tokenizer = hf_tokenizer
        self.sampling = sampling

    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        device, dtype = a_tensor.device, a_tensor.dtype
        if self.sampling=='fp32_gumbel': dtype = torch.float32
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

    def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        sentA_lenths (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs, sentA_lenths)
        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0] # (B, L, vocab size)
        # reduce size to save space and speed
        mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)

        with torch.no_grad():
            # sampling
            pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
            # produce inputs for discriminator
            generated = masked_inputs.clone() # (B,L)
            generated[is_mlm_applied] = pred_toks # (B,L)
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone() # (B,L)
            is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.hf_tokenizer.pad_token_id
        seq_len = input_ids.shape[1]
        token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],  
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

class LightningModule(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        # Init hyperparameters
        self.save_hyperparameters(config)
        
        ## model
        gen_config = ElectraConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_size=self.hparams.embedding_dim,
            hidden_size=256,
            num_hidden_layers=self.hparams.n_layers//4,
            num_attention_heads=self.hparams.n_heads,
            intermediate_size=128,
            hidden_act="gelu",
            hidden_dropout_prob=self.hparams.dropout,
            attention_probs_dropout_prob=self.hparams.dropout,
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
            hidden_act="gelu",
            hidden_dropout_prob=self.hparams.dropout,
            attention_probs_dropout_prob=self.hparams.dropout,
            max_position_embeddings=512,
            type_vocab_size=2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,)
        
        generator = ElectraForMaskedLM(gen_config)
        discriminator = ElectraForPreTraining(disc_config)
        discriminator.electra.embeddings = generator.electra.embeddings
        generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

        self.electra_model = ELECTRAModel(generator, discriminator, tokenizer)
        

def filter_unwanted_tokens(tensor, tokens, ignore_tokens, tokenizer):
    wanted_idx, wanted_tokens = [], []
    ignore_tokens = tokenizer.encode(ignore_tokens, add_special_tokens=False)
    tokens = tokenizer.encode(tokens, add_special_tokens=False)
    for idx, token in enumerate(tokens):
        if token in ignore_tokens:
            continue
        else:
            wanted_idx.append(idx)
            wanted_tokens.append(token)

    wanted_tokens = tokenizer.convert_ids_to_tokens(wanted_tokens)
    return tensor[wanted_idx, :][:, wanted_idx], wanted_tokens

def get_attentions(module, inputs, length_mask, token_type_ids):
    result = module.electra_model.discriminator(inputs, length_mask, token_type_ids, output_attentions=True)
    return result.attentions  # a tuple, length is equal to attention layers num

def visualize_attention(hparams, attentions, smiles, img_path='./img/test.svg', tokenizer=None):
    ignore_tokens = ["(", ")", "=", "#", "-", '.', "/", '\\', "@"]
    ignore_tokens += ["1", "2", '3', '4', '5', '6','7','8','9','%10']
    ignore_tokens += ["<eos>", "<bos>", "<pad>", "<mask>"]
    
    fig, axarr = plt.subplots(2,hparams['n_layers']//2,figsize=(30, 15))
    
    for layer_idx in range(len(attentions)):
        some_layer = attentions[layer_idx].squeeze()
        tensor = torch.mean(some_layer, dim=0).cpu()
        
        tensor = ((tensor+torch.transpose(tensor, 0, 1))/2)
        # tensor.fill_diagonal_(0.0)
        tensor = tensor.detach().numpy() 
        
        tensor, filtered_tokens = filter_unwanted_tokens(tensor, smiles, ignore_tokens, tokenizer)
        
        axarr[layer_idx%2, layer_idx//2].imshow(tensor, cmap='viridis', interpolation='none', aspect='equal')

        axarr[layer_idx%2, layer_idx//2].set_xticks(range(0, len(filtered_tokens)))
        axarr[layer_idx%2, layer_idx//2].set_yticks(range(0, len(filtered_tokens)))

        axarr[layer_idx%2, layer_idx//2].set_yticklabels(filtered_tokens)
        axarr[layer_idx%2, layer_idx//2].set_xticklabels(filtered_tokens)

        axarr[layer_idx%2, layer_idx//2].set_title(f"Avg-Pooled Heads / Layer = {layer_idx+1}")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3)
    plt.savefig(img_path, format='svg')

def visualize_one_attention(hparams, attentions, smiles, img_path='./img/test.svg'):
    ignore_tokens = ["(", ")", "=", "#", "-", '.']
    ignore_tokens += ["1", "2", '3', '4', '5', '6','7','8','9','%10']
    ignore_tokens += ["<eos>", "<bos>", "<pad>", "<mask>"]
    
    fig, axarr = plt.subplots(2,hparams['n_layers']//2,figsize=(30, 15))
    shape = attentions[0].shape
    result = torch.zeros((shape[2], shape[3]))
    print(result.shape)
    for layer_idx in range(len(attentions)):
        some_layer = attentions[layer_idx].squeeze()
        tensor = torch.mean(some_layer, dim=0).cpu()
        
        tensor = ((tensor+torch.transpose(tensor, 0, 1))/2)
        # tensor.fill_diagonal_(0.0)
        result += tensor
        
        tensor = result.detach().numpy() 
        
        tensor, filtered_tokens = filter_unwanted_tokens(tensor, smiles, ignore_tokens)
        
        axarr[layer_idx%2, layer_idx//2].imshow(tensor, cmap='viridis', interpolation='none', aspect='equal')

        axarr[layer_idx%2, layer_idx//2].set_xticks(range(0, len(filtered_tokens)))
        axarr[layer_idx%2, layer_idx//2].set_yticks(range(0, len(filtered_tokens)))

        axarr[layer_idx%2, layer_idx//2].set_yticklabels(filtered_tokens)
        axarr[layer_idx%2, layer_idx//2].set_xticklabels(filtered_tokens)

        axarr[layer_idx%2, layer_idx//2].set_title(f"Avg-Pooled Heads / Layer = {layer_idx+1}")

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3)
    plt.savefig(img_path, format='svg')

def visualize_head(hparams, attentions, smiles, img_path='./img/test.svg', tokenizer=None):
    img_path = img_path.replace(".svg", "_head.svg")
    ignore_tokens = ["(", ")", "=", "#", "-", '.', "/", '\\', "@"]
    ignore_tokens += ["1", "2", '3', '4', '5', '6','7','8','9','%10']
    ignore_tokens += ["<eos>", "<bos>", "<pad>", "<mask>"]
    num_heads = attentions[0].shape[1]
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads*5, 5))

    # Aggregate attention across all layers for each head
    aggregate_attention = [torch.zeros_like(attentions[0][0, 0]) for _ in range(num_heads)]
    for layer_attention in attentions:
        for j in range(num_heads):
            aggregate_attention[j] += layer_attention[0, j]
    
    for j in range(num_heads):
        ax = axes[j]
        some_layer = attentions[j].squeeze()
        tensor = torch.mean(some_layer, dim=0).cpu()
        
        tensor = ((tensor+torch.transpose(tensor, 0, 1))/2)
        # tensor.fill_diagonal_(0.0)
        head_attention = tensor.detach().numpy() / len(attentions)
        
        # head_attention = aggregate_attention[j].cpu().detach().numpy() / len(attentions)
        filtered_attention, filtered_tokens = filter_unwanted_tokens(head_attention, smiles, ignore_tokens, tokenizer)
        ax.imshow(filtered_attention, cmap='viridis', interpolation='none', aspect='equal')
        ax.set_xticks(range(len(filtered_tokens)))
        ax.set_yticks(range(len(filtered_tokens)))
        ax.set_xticklabels(filtered_tokens)
        ax.set_yticklabels(filtered_tokens)
        ax.set_title(f'Head {j+1}')

    plt.tight_layout()
    plt.savefig(img_path)
    
def visualize_aggregate(hparams, attentions, smiles, img_path='./img/test.svg', tokenizer=None, title=None):
    img_path = img_path.replace(".svg", "_aggregated.svg")
    ignore_tokens = ["(", ")", "=", "#", "-", '.', "/", '\\', "@"]
    ignore_tokens += ["1", "2", '3', '4', '5', '6','7','8','9','%10']
    ignore_tokens += ["<eos>", "<bos>", "<pad>", "<mask>"]
    num_heads = attentions[0].shape[1]
    num_layers = len(attentions)
    
    # Initialize an aggregated attention tensor
    aggregate_attention = torch.zeros_like(attentions[0][0, 0])
    
    for layer_attention in attentions:
        # layer_attention = torch.nn.functional.normalize(layer_attention, p=1, dim=0)
        aggregate_attention += layer_attention.sum(dim=0)[0]

    # Calculate the average attention across layers
    # aggregate_attention /= num_layers
    aggregate_attention = ((aggregate_attention+torch.transpose(aggregate_attention, 0, 1))/2)
    
    # aggregate_attention = torch.nn.functional.normalize(aggregate_attention, p=1, dim=0)
    # aggregate_attention[aggregate_attention < 0.05] = 0
    
    # aggregate_attention.fill_diagonal_(0.0)
    
    # Filter unwanted tokens and get tokens list
    filtered_attention, filtered_tokens = filter_unwanted_tokens(aggregate_attention.cpu().detach().numpy(), smiles, ignore_tokens, tokenizer)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(filtered_attention, cmap='viridis', interpolation='none', aspect='equal')
    ax.set_xticks(range(len(filtered_tokens)))
    ax.set_yticks(range(len(filtered_tokens)))
    ax.set_xticklabels(filtered_tokens)
    ax.set_yticklabels(filtered_tokens)
    ax.set_title(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(img_path)

    
def get_img(smiles, module, tokenizer, hparams, img_path, device, title):
    tokens = tokenizer.batch_encode_plus([smile for smile in smiles],
                                         padding=True,
                                         add_special_tokens=True,
                                         return_special_tokens_mask=True)
    smiles = smiles[0]
    inputs = torch.tensor(tokens["input_ids"]).to(device)
    length_mask = torch.tensor(tokens["attention_mask"]).to(device)
    token_type_ids = torch.tensor(tokens["token_type_ids"]).to(device)

    # Get attentions and draw it
    attentions = get_attentions(module, inputs, length_mask, token_type_ids)
    # visualize_attention(hparams, attentions, smiles, img_path=img_path, tokenizer=tokenizer)
    # visualize_head(hparams, attentions, smiles, img_path=img_path, tokenizer=tokenizer)
    visualize_aggregate(hparams, attentions, smiles, img_path=img_path, tokenizer=tokenizer, title=title)
    # visualize_one_attention(hparams, attentions, smiles, img_path=img_path)
    

def main():
    ## QT
    path = "/path/to/checkpoints/"
    checkpoint = path+"checkpoints/epoch=399-step=1200.ckpt"
    
    hparams_file = path+"hparams.yaml"
    smiles = {
        "chloroquine": "CCN(CC)CCCC(C)NC1=CC=NC2=CC(Cl)=CC=C12",
        "hydroxychloroquine": "CCN(CCO)CCCC(C)NC1=CC=NC2=CC(Cl)=CC=C12",
    }
    img_path = "./img/attention_qt_test/"
    os.mkdir(img_path)
    # Prepare
    with open(hparams_file, 'r') as file:
        hparams = yaml.safe_load(file)
    device = hparams['device']
    
    tokenizer = Tokenizer("bert_vocab_qt.txt")
    module = LightningModule.load_from_checkpoint(
        checkpoint,
        hparams_file=hparams_file,
        config=hparams, 
        tokenizer=tokenizer,)
    
    for key in smiles:
        path = img_path + ("{0}.svg".format(key))
        get_img([smiles[key]], module, tokenizer, hparams, path, device, title=key)
    
    
    
    
    # RA
    path = "/path/to/checkpoints/"
    checkpoint = path+"checkpoints/epoch=399-step=1200.ckpt"
    
    hparams_file = path+"hparams.yaml"
    smiles = {
        "cerivastatin":"COCc1c(C(C)C)nc(C(C)C)c(C=CC(O)CC(O)CC(=O)O)c1-c1ccc(F)cc1",
        "lovastatin":     "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
    }
    img_path = "./img/attention_ra/"
    
    # Liver
    path = "/path/to/checkpoints/"
    checkpoint = path+"checkpoints/epoch=399-step=2000.ckpt"
    hparams_file = path+"hparams.yaml"
    smiles = {
        "lbuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "lbufenac" : "CC(C)Cc1ccc(CC(=O)O)cc1",
        
        "zolpidem" : "CN(C)C(=O)CC1=C(N=C2N1C=C(C=C2)C)C3=CC=C(C=C3)C",
        "alpidem"  : "CCCN(CCC)C(=O)CC1=C(N=C2 N1C=C(C=C2)Cl)C3=CC=C(C=C3)Cl",
    }
    img_path = "./img/attention_li_test/"
    
    # Prepare
    with open(hparams_file, 'r') as file:
        hparams = yaml.safe_load(file)
    device = hparams['device']
    os.makedirs(img_path, exist_ok=True)
    tokenizer = Tokenizer("bert_vocab_qt.txt")
    module = LightningModule.load_from_checkpoint(
        checkpoint,
        hparams_file=hparams_file,
        config=hparams, 
        tokenizer=tokenizer,)
    
    # fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
    for key in smiles:
        path = img_path + ("{0}.svg".format(key))
        get_img([smiles[key]], module, tokenizer, hparams, path, device, title=key)
    

if __name__ == "__main__":
    main()

