from transformers import BertTokenizer
import regex as re

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class Tokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string
    


if __name__ == "__main__":
    tokenizer = Tokenizer("bert_vocab.txt")
    "C#CC1(CCCCC1)OC(=O)N", 
    "CC1=C(SC(=N1)C2=CC(=C(C=C2)OCC(C)C)C#N)C(=O)O",
    "C1CN(CCC1NC(=O)C2=CC=CC=C2C3=CC=C(C=C3)C(F)(F)F)CCCCC4(C5=CC=CC=C5C6=CC=CC=C64)C(=O)NCC(F)(F)F"
    
    smiles = ["CCCmaskCCC"]
    
    tokens = tokenizer.batch_encode_plus([smile for smile in smiles], 
                                         padding=True, 
                                         add_special_tokens=True,
                                         return_special_tokens_mask=True)
    
    print(tokens['input_ids'])
    print(tokens['attention_mask'])
    print(tokenizer.batch_decode(tokens['input_ids'], skip_special_tokens=False))
    print(tokens)