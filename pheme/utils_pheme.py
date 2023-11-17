import argparse
import logging
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'],
                        help='mode')
    # encoder config
    parser.add_argument('--posts_path', default='./pheme_data', type=str,
                        help='train text file')
    parser.add_argument('--image_path', default='./pheme_data/images', type=str,
                        help='train image folder')

    parser.add_argument('--data_file', default='./pheme_data/pheme_data.pkl', type=str,
                        help='path to save data file')
    parser.add_argument('--d_model_t', default=768, type=int,
                        help='d_model of transformer encoder layer')
    parser.add_argument('--nhead_t', default=4, type=int,
                        help='nhead of transformer encoder layer')
    parser.add_argument('--num_layers_t', default=1, type=int,
                        help='num_layers of transformer encoder layer')
    parser.add_argument('--nhead_c_s', default=4, type=int,
                        help='nhead of cross attention and self attention layer')
    parser.add_argument('--hidden_size_c_s', default=768, type=int,
                        help='output dim of cross attention and self attention layer')
    parser.add_argument('--hidden_dropout_prob', default=0.4, type=float,
                        help='dropout rate in attention output layer')
    parser.add_argument('--activation', default='gelu', type=str, choices=['gelu', 'relu'],
                        help='activation function in attention module')
    parser.add_argument('--num_c_layer', default=1, type=int,
                        help='number of cross module encoder')
    parser.add_argument('--hidden_size_linear', default=128, type=int,
                        help='hidden size of output linear layer')
    parser.add_argument('--itm_head', default='./weights/itm_head.pth', type=str,
                        help='itm_head parameters')
    parser.add_argument('--hidden_size_itm', default=48, type=int,
                        help='output dim of itm')
    parser.add_argument('--hidden_size_input', default=128, type=int,
                        help='output dim of input embedding')

    # train config
    parser.add_argument('--max_len', default=512, type=int,
                        help='max token len')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch_size')
    parser.add_argument('--data_seed', default=2, type=int,
                        help='Random number seeds for partitioning data sets')
    parser.add_argument('--seed', default=2, type=int,
                        help='Random number seed')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='learning rate')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint')
    parser.add_argument('--epochs', default=30, type=int,
                        help='epochs')
    parser.add_argument('--theta', default=0.01, type=float,
                        help='the proportion of divergence loss in total loss')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequence')
    parser.add_argument('--device', default='gpu', type=str,
                        help='device')
    parser.add_argument('--beta', default=0.8, type=float,
                        help='the proportion of text loss in classification loss')

    args = parser.parse_args()
    return args


def device(config):
    if torch.cuda.is_available() and config.device == 'gpu':
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    return dev


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class Optimizer(object):
    def __init__(self, args, optimizer, warmup_steps=1000):
        self.optimizer = optimizer
        self.init_lr = args.d_model_t ** (-0.5)
        self.warmup_steps = warmup_steps
        self.lr = self.init_lr
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        self.min_lr = 1e-5
        self.lr = self.init_lr * min(self.step_num ** (-0.65), self.step_num * (self.warmup_steps ** (-1.5)))
        self.lr = max(self.lr, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


def get_extended_attention_mask(attention_mask):
    '''
    make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    modified from
    https://huggingface.co/transformers/_modules/transformers/modeling_utils.html#ModuleUtilsMixin.get_extended_attention_mask
    '''
    assert attention_mask.dim() == 2
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
