'''
create transformer and cross-attention model
'''
import math
import torch
from torch import nn
import torch.nn.functional as F

from utils_twitter import get_extended_attention_mask


class MutualModel(nn.Module):
    def __init__(self, config):
        super(MutualModel, self).__init__()
        self.text_encoder = Transformer(config)
        self.visual_encoder = Transformer(config)
        self.cross_module = CrossModule(config)
        self.pos_embedding = PositionalEncoding(d_model=config.d_model_t,
                                                max_len=config.max_len)

        self.text_linear_1 = nn.Linear(config.d_model_t, config.hidden_size_linear)
        self.text_linear_2 = nn.Linear(config.hidden_size_linear, 2)

        self.visual_linear_1 = nn.Linear(config.d_model_t, config.hidden_size_linear)
        self.visual_linear_2 = nn.Linear(config.hidden_size_linear, 2)

    def get_pred(self, text_output, visual_output):
        text_output = text_output.mean(1)
        visual_output = visual_output.mean(1)

        text_output = self.text_linear_2(self.text_linear_1(text_output))
        visual_output = self.visual_linear_2(self.visual_linear_1(visual_output))

        return text_output, visual_output

    def forward(self, text_feat, text_attention_mask, visual_feat, visual_attention_mask, vilt_embedding):
        pos_embed_text = self.pos_embedding(text_feat)
        pos_embed_visual = self.pos_embedding(visual_feat)
        text_output = pos_embed_text + text_feat
        visual_output = pos_embed_visual + visual_feat

        # transformer encoder
        text_out = self.text_encoder(text_output)
        visual_out = self.visual_encoder(visual_output)

        # cross module encoder
        text_out, visual_out = self.cross_module(text_out, text_attention_mask,
                                                 visual_out, visual_attention_mask,
                                                 vilt_embedding)

        text_out, visual_out = self.get_pred(text_out, visual_out)
        return text_out, visual_out


class ITM(nn.Module):
    def __init__(self, config):
        super(ITM, self).__init__()
        self.pretrained_itm = torch.load(config.itm_head)
        self.matching_layer = nn.Linear(self.pretrained_itm.out_features, config.hidden_size_itm)

    def forward(self, hidden_states_matching):
        matching_score = self.pretrained_itm(hidden_states_matching)
        matching_embed = F.relu(self.matching_layer(matching_score))
        return matching_embed


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model_t,
                                                   nhead=config.nhead_t, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=config.num_layers_t)

    def forward(self, input_embedding):
        return self.encoder(input_embedding)


class CrossModule(nn.Module):
    def __init__(self, config):
        super(CrossModule, self).__init__()
        self.cross_encoder = nn.ModuleList(
            [CrossModuleLayer(config) for _ in range(config.num_c_layer)]
        )

    def forward(self, text_feat, text_attention_mask, visual_feat, visual_attention_mask, vilt_embedding):
        text_out, visual_out = text_feat, visual_feat
        for cross_layer in self.cross_encoder:
            text_out, visual_out = cross_layer(text_out, text_attention_mask, visual_out, visual_attention_mask,
                                               vilt_embedding)

        return text_out, visual_out


class PositionalEncoding(nn.Module):
    """
    Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


class CrossModuleLayer(nn.Module):
    '''
    https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
    '''

    def __init__(self, config):
        super(CrossModuleLayer, self).__init__()
        # cross attention layer
        self.crossatt_based_text = CrossAttLayer(config, 512)
        self.crossatt_based_visual = CrossAttLayer(config, 196)

        self.itm_head = ITM(config)

        # text_based attention module
        self.text_selfatt = SelfAttLayer(config)
        self.text_inter = Intermediate(config)
        self.text_output = FFNOutput(config)

        # visual_based attention module
        self.visual_selfatt = SelfAttLayer(config)
        self.visual_inter = Intermediate(config)
        self.visual_output = FFNOutput(config)

    def cross_att_layer(self, text_input, text_attention_mask, visual_input, visual_attention_mask, matching_embed):
        text_based_output = self.crossatt_based_text(text_input, visual_input, matching_embed,
                                                     crossatt_mask=visual_attention_mask)
        visual_based_output = self.crossatt_based_visual(visual_input, text_input, matching_embed,
                                                         crossatt_mask=text_attention_mask)
        return text_based_output, visual_based_output

    def self_att_layer(self, text_input, text_attention_mask, visual_input, visual_attention_mask):
        text_att_output = self.text_selfatt(text_input, text_attention_mask)
        visual_att_output = self.visual_selfatt(visual_input, visual_attention_mask)
        return text_att_output, visual_att_output

    def ffn_layer(self, text_input, visual_input):
        text_inter_output = self.text_inter(text_input)
        visual_inter_output = self.visual_inter(visual_input)

        text_out = self.text_output(text_inter_output, text_input)
        visual_out = self.visual_output(visual_inter_output, visual_input)
        return text_out, visual_out

    def forward(self, text_feat, text_attention_mask, visual_feat, visual_attention_mask, vilt_embedding):
        text_att_output = text_feat
        visual_att_output = visual_feat

        matching_embed = self.itm_head(vilt_embedding)

        text_att_output, visual_att_output = self.cross_att_layer(text_att_output, text_attention_mask,
                                                                  visual_att_output, visual_attention_mask,
                                                                  matching_embed)

        text_att_output, visual_att_output = self.self_att_layer(text_att_output, text_attention_mask,
                                                                 visual_att_output, visual_attention_mask)
        text_out, visual_out = self.ffn_layer(text_att_output, visual_att_output)

        return text_out, visual_out


class Attention(nn.Module):
    '''
    Created and modified from BERT
    '''

    def __init__(self, config, ctx_dim=None):
        super(Attention, self).__init__()
        if config.hidden_size_c_s % config.nhead_c_s != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size_c_s, config.nhead_c_s))
        self.nhead_c_s = config.nhead_c_s
        self.attention_head_size = int(config.hidden_size_c_s / config.nhead_c_s)
        self.all_head_size = self.nhead_c_s * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size_c_s
        self.query = nn.Linear(config.hidden_size_c_s, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.nhead_c_s, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, tensor_q, tensor_k_v, attention_mask=None):
        mixed_query_layer = self.query(tensor_q)
        mixed_key_layer = self.key(tensor_k_v)
        mixed_value_layer = self.value(tensor_k_v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            extended_attention_mask = get_extended_attention_mask(attention_mask)
            attention_scores = attention_scores + extended_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


def gelu(x):
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size_c_s, config.hidden_size_c_s)
        self.intermediate_act_fn = ACT2FN[config.activation]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SelfAttLayer(nn.Module):
    def __init__(self, config):
        super(SelfAttLayer, self).__init__()
        self.att = Attention(config)
        self.out = AttOutput(config)

    def forward(self, input_tensor_q, attention_mask):
        att_output = self.att(input_tensor_q, input_tensor_q, attention_mask)
        ffn_output = self.out(att_output, input_tensor_q)
        return ffn_output


class UpdateGate(nn.Module):
    def __init__(self, config, upgate_dim):
        super(UpdateGate, self).__init__()
        self.update = nn.Linear(config.hidden_size_itm, upgate_dim)

    def forward(self, matching_embed):
        update_weight = F.sigmoid(self.update(matching_embed))
        return update_weight


class CrossAttLayer(nn.Module):
    def __init__(self, config, upgate_dim):
        super(CrossAttLayer, self).__init__()
        self.att = Attention(config)
        self.upgate = UpdateGate(config, upgate_dim)
        self.out = AttOutput(config)

    def forward(self, input_tensor_q, input_tensor_k_v, matching_embed, crossatt_mask=None):
        att_output = self.att(input_tensor_q, input_tensor_k_v, crossatt_mask)

        sequence_weight = self.upgate(matching_embed)

        update_gate_output = torch.zeros_like(att_output)
        for i in range(update_gate_output.shape[0]):
            update_gate_output[:, i, :] = sequence_weight[:, :, i] * att_output[:, i, :]

        ffn_output = self.out(update_gate_output, input_tensor_q)
        return ffn_output


class AttOutput(nn.Module):
    def __init__(self, config):
        super(AttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size_c_s, config.hidden_size_c_s)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size_c_s, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FFNOutput(nn.Module):
    def __init__(self, config):
        super(FFNOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size_c_s, config.hidden_size_c_s)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size_c_s, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
