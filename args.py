import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--skip_special_tokens', type=bool, default=True, help='Skip special tokens (bos, eos, pad) when predicting')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1, default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2, default=0.999')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--decay_steps', type=int, nargs='+', default=None, help='when to decay the learning rate')
    parser.add_argument('--decay_gamma', type=float, default=0.1, help='decay factor for learning rate')
    # 
    parser.add_argument('--n_heads',
                           type=int, default=8,
                           help='number of heads')
    parser.add_argument('--n_layers',
                           type=int, default=12,
                           help='GPT number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, required=False, help='dropout')

    parser.add_argument('--en_dropout',
                        type=float, default=0.1,
                        help='Encoder layers dropout')
    parser.add_argument('--embedding_dim',
                           type=int, default=128,
                           help='Latent vector dimensionality')
    parser.add_argument('--alpha',
                           type=float, default=1.0,
                           help='Coefficient of KL for RDrop')
    parser.add_argument("--mask_prob", type=float, default=0.4, help='Ratio for masking token', required=False)
    # 
    parser.add_argument('--valid_every', type=int, default=2, help='valid every x epochs')
    parser.add_argument('--checkpoint_every',
                           type=int, default=1000,
                           help='save checkpoint every x iterations')
    parser.add_argument('--checkpoint_path',
                           type=str, default='',
                           help='checkpoint path')
    parser.add_argument('--device',
                        type=str, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=1104,
                        help='Random seed')
    parser.add_argument('--max_epochs', type=int, required=False, default=1, help='max number of epochs')
    parser.add_argument("--train_dataset_length", type=int, default=None, required=False)
    parser.add_argument("--eval_dataset_length", type=int, default=None, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--num_classes", type=int, required=False)
    parser.add_argument("--dataset_name", type=str, required=False, default="sol")
    parser.add_argument("--measure_name", type=str, required=False, default="label", help="name for label")
    parser.add_argument("--checkpoints_folder", type=str, required=True)
    parser.add_argument("--checkpoint_root", type=str, required=False)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--isomericSmiles", action='store_true', required=False)
    parser.add_argument("--randomize_smiles", action='store_true', required=False)
    parser.add_argument("--disc_smooth_label", action='store_true', required=False)
    parser.add_argument("--gen_smooth_label", action='store_true', required=False)
    parser.add_argument("--freeze", action='store_true', required=False)
    parser.add_argument("--sampling", type=str, required=False, choices=["fp32_gumbel", "fp16_gumbel", "multinomial"], default="multinomial")

    parser.add_argument("--scheduler", 
                        default="CosineAnnealingLR", 
                        required=False, 
                        choices=["MultiStepLR", "CosineAnnealingLR"], 
                        help="scheduler")
    parser.add_argument("--optimizer", 
                        default="AdamW", 
                        required=False, 
                        choices=["AdamW", "RMSprop", "SGD"], 
                        help="optimizer")
    parser.add_argument('--loss', type=str, default="mean", choices=['mean', 'sum'],required=False,)
    parser.add_argument('--warmup', type=int, default=0 , required=False,)
    parser.add_argument('--lower', type=int, required=False, default=1, help='size for span masking')    
    parser.add_argument('--upper', type=int, required=False, default=10, help='size for span masking')    
    parser.add_argument('--geometric_p', type=float, required=False, default=.2, help='geometric prob for span masking')    
    parser.add_argument('--mask', 
                        type=str, 
                        required=False, 
                        default='token', 
                        choices=['span', 'token'], 
                        help='mask strategy')                  
    parser.add_argument('--replacement', 
                        type=str, 
                        required=False, 
                        default='span', 
                        choices=['span', 'token'], 
                        help='replacement strategy for span masking') 
    parser.add_argument('--threshold', 
                        type=float, 
                        required=False, 
                        default=0.1, 
                        help='threshold for detect QT drugs in validation step') 
    parser.add_argument('--hparams_file', 
                        type=str, 
                        required=False, 
                        help='path hparams_file')
    parser.add_argument('--hidden_size', 
                        type=int, 
                        default=256,
                        required=False, 
                        help='hidden size for electra')
    parser.add_argument('--task',
                        type=str, 
                        default='classification',
                        choices=['regression', 'classification', 'binary_classification'],
                        required=False, 
                        help='task for model')

    # Downstream
    parser.add_argument('--downstream_model',
                        required=False,
                        choices=['CrossAttention', 'SelfAttention'],
                        default='CrossAttention',)
    parser.add_argument('--cls_type', 
                        required=False,
                        choices=['sum', 'first'],
                        default='first',
                        help="How to deal with embeds for downstream")
    parser.add_argument('--depth',
                        type=int, 
                        default=2,
                        help='number of layers of downstream model')
    parser.add_argument('--drop_norm',
                        type=float, 
                        default=0.0,
                        required=False,
                        help='number for drop norm')
    parser.add_argument('--scale', 
                        type=float, 
                        default=0.125, 
                        required=False, 
                        help='Scale for FlashAttention')
    parser.add_argument('--complex', 
                        action='store_false',
                        required=False, 
                        help='embedding type for info')
    parser.add_argument('--accumulate_grad_batches', 
                        type=int,
                        required=False, 
                        default=16,
                        help='embedding type for info')

    
    return parser

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args