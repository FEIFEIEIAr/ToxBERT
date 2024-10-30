import torch
import torch.utils.data
import numpy as np
from pandarallel import pandarallel

from util import *
from constant import BOS_ID, EOS_ID, PAD_ID, MASK_ID, IGN_ID

pandarallel.initialize()


class MaskScheme():
    def __init__(self, mlm_probability=0.4, tokenizer=None) -> None:
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        self.char2id = {"<bos>":BOS_ID, "<eos>":EOS_ID, "<pad>":PAD_ID, "<mask>":MASK_ID}
    
    def __call__(self, inputs, special_tokens_mask=None):
        return self.mask(inputs, special_tokens_mask=None)

class TokenMask(MaskScheme):
    def __init__(self, mlm_probability=0.4, tokenizer=None) -> None:
        super().__init__(mlm_probability, tokenizer)
    
    def mask(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        Args:
            inputs (Tensor): Inputs ids, shape (batch_size, seq_len). From transformer.tokenizer.batch_encode_plus.
            special_tokens_mask (Tensor): Mask for special tokens. From transformer.tokenizer.batch_encode_plus
            
        Returns:
            Tuple: 
                inputs (Tensor): Inputs ids, shape (batch_size, seq_len).
                labels (Tensor): Labels ids, shape (batch_size, seq_len).
                masked_indices (Tensor): Indices of mask indicate tokens that will be calc loss.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # print("Mask prob:", self.mlm_probability)
        probability_matrix = torch.full(labels.size(), self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = IGN_ID  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        inputs[indices_replaced] = MASK_ID

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.size(), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.char2id.keys()), self.tokenizer.vocab_size, labels.size(), dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices
    
class SpanMaskScheme(MaskScheme):
    def __init__(self, 
                 mlm_probability=0.4,
                 tokenizer=None, 
                 lower=1, 
                 upper=10, 
                 replacement='span',
                 geometric_p=0.2) -> None:
        super().__init__(mlm_probability, tokenizer)
        self.lower = lower  # Smallest span length
        self.upper = upper  # Largest span length
        self.replacement = replacement
        self.lens = list(range(self.lower, self.upper + 1))
        self.p = geometric_p
        self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        print("Span distribution{}. Span length{}.".format(self.len_distrib, self.lens))
        
class NERSpanMask():
    def __init__(self, ) -> None:
        pass
        
class SpanMask(SpanMaskScheme):
    def __init__(self, 
                 mlm_probability=0.4,
                 tokenizer=None, 
                 lower=1, 
                 upper=10, 
                 replacement='span',
                 geometric_p=0.2):
        super().__init__(mlm_probability, tokenizer, lower, upper, replacement, geometric_p)
    
    def mask(self, inputs, special_tokens_mask=None):
        """Span masking for a batch of inputs
 
        Args:
            inputs (Tensor): 2d tensor of input ids, [batch_size, seq_len]
            special_tokens_mask (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor: inputs: masked input ids, dtype: long, [batch_size, seq_len]
                    labels: original tokens for masked position, dtype: long, [batch_size, seq_len]
                    masked_indices: indication for masked position, dtype: bool, [batch_size, seq_len]
        """
        labels = inputs.clone()
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor(special_tokens_mask).bool()

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for i in range(len(inputs)):
            inputs[i], masked_indices[i] = self.mask_sentence(inputs[i], special_tokens_mask[i])
            
        labels[~masked_indices] = IGN_ID
        return inputs, labels, masked_indices
    
    def mask_sentence(self, inputs, special_tokens_mask):
        """Span masking for a sentence

        Args:
            inputs (Tensor): 1d tensor of input ids, [seq_len]
            special_tokens_mask (Tensor): indiction for special tokens. [seq_len]

        Returns:
            Tensor: inputs: masked input ids, dtype: long, [seq_len]
                    labels: original tokens for masked position, dtype: long, [seq_len]
                    masked_indices: indication for masked position, dtype: bool, [seq_len]
        """
        
        self.seq_length = (~special_tokens_mask).sum()

        # We fix num of masked tokens to 15% of total tokens per batch 
        self.mask_num = torch.ceil(self.seq_length * self.mlm_probability)
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        self.spans = []

        while masked_indices.sum() < self.mask_num:
            inputs, masked_indices = self.mask_span(inputs, masked_indices, special_tokens_mask)
        return inputs, masked_indices
    
    def mask_span(self, inputs, masked_indices, special_tokens_mask):
        span_len = np.random.choice(self.lens, p=self.len_distrib)
        span_start  = np.random.choice(self.seq_length)+1
        span_end = span_start + span_len
        
        # Break out conditions
        # span covers masked tokens
        if torch.any(masked_indices[span_start: span_end]):
            return inputs, masked_indices
        # span covers special tokens. also work for [sep]...
        if torch.any(special_tokens_mask[span_start: span_end]):
            return inputs, masked_indices
        # mask num is reached
        if span_len + masked_indices.sum() > self.mask_num:
            return inputs, masked_indices
        
        masked_indices[span_start: span_end] = True
        if self.replacement == 'span':
            rand = np.random.random()
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            if rand < 0.8:
                inputs[span_start: span_end] = MASK_ID
            # 10% of the time, we replace masked input tokens with random word
            elif rand < 0.9:
                inputs[span_start: span_end] = torch.randint(len(self.char2id.keys()), self.tokenizer.vocab_size, (span_len, ), dtype=torch.long)
            # The rest of the time (10% of the time) we keep the masked input tokens unchanged    
        elif self.replacement == 'token':
            for i in range(span_start, span_end):
                rand = np.random.random()
                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                if rand < 0.8:
                    inputs[i] = MASK_ID
                # 10% of the time, we replace masked input tokens with random word
                elif rand < 0.9:
                    inputs[i] = torch.randint(len(self.char2id.keys()), self.tokenizer.vocab_size, (1, ), dtype=torch.long)
                # The rest of the time (10% of the time) we keep the masked input tokens unchanged    
        else:
            raise ValueError("Invalid replacement method! Only support 'span' or 'token' level replacement.")
        
        return inputs, masked_indices
        