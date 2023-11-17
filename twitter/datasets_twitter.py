import pickle
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import transformers
from PIL import Image

from utils_twitter import get_logger
from utils_twitter import device

logger = get_logger()


class DataIterater(object):
    def __init__(self, datas, config):
        self.batch_size = config.batch_size
        self.datas = datas
        self.max_len = config.max_len
        self.num_batches = len(datas) // self.batch_size
        self.residue = False
        if len(datas) % (self.num_batches * self.batch_size) != 0:
            self.residue = True
        self.index = 0
        self.device = device(config=config)

        # text embedding model
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-cased')
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

        # image embedding model
        self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit_model = torch.nn.Sequential(*list(self.vit_model.children())[:-3])

        self.bert_model.to(self.device)
        self.vit_model.to(self.device)

        self.bert_model.eval()
        self.vit_model.eval()

    def _to_tensor(self, datas):
        x_texts, x_texts_mask, x_images, x_images_mask, matching_embed = [], [], [], [], []
        y = []
        for sample in datas:
            x_text, x_text_mask = self.get_text_embedding(sample['text'])
            x_image, x_image_mask = self.get_image_embedding_ViT(sample['image'])
            matching = sample['embedding']

            x_texts.append(x_text)
            x_texts_mask.append(x_text_mask)
            x_images.append(x_image)
            x_images_mask.append(x_image_mask)
            matching_embed.append(matching)
            y.append(sample['label'])

        x_texts = torch.stack(x_texts)
        x_texts_mask = torch.stack(x_texts_mask)
        x_images = torch.stack(x_images)
        x_images_mask = torch.stack(x_images_mask)
        matching_embedding = torch.stack(matching_embed).to(self.device)
        y = torch.LongTensor(y).to(self.device)

        return (x_texts, x_texts_mask, x_images, x_images_mask, matching_embedding), y

    def __next__(self):
        if self.residue and self.index == self.num_batches:
            data = self.datas[self.index * self.batch_size: len(self.datas)]
            self.index += 1
            data = self._to_tensor(data)
            return data
        elif self.index > self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            data = self.datas[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            data = self._to_tensor(data)
            return data

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches

    def get_image_embedding_ViT(self, image_root):
        data_augment_config = {'input_size': (3, 224, 224),
                               'interpolation': 'bicubic',
                               'mean': (0.4406, 0.4233, 0.4081),
                               'std': (0.2402, 0.2352, 0.2300),
                               'crop_pct': 0.9}
        data_config = resolve_data_config(data_augment_config, model=self.vit_model)
        transform = create_transform(**data_config)
        img = Image.open(image_root).convert('RGB')
        img = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.vit_model(img)

        attn_mask = torch.ones([image_embedding.shape[0], image_embedding.shape[1]], device=self.device)

        return image_embedding.squeeze(0), attn_mask.squeeze(0)

    def get_text_embedding(self, text):
        token = self.tokenizer.tokenize(text)
        if len(token) > 510:
            token = token[:510]
        token = ['[CLS]'] + token + ['[SEP]']
        padded_token = token + ['[PAD]' for _ in range(self.max_len - len(token))]
        attn_mask = [1 if token != '[PAD]' else 0 for token in padded_token]
        seg_ids = [0 for _ in range(len(padded_token))]
        token_ids = self.tokenizer.convert_tokens_to_ids(padded_token)
        token_ids = torch.tensor(token_ids).unsqueeze(0).to(self.device)
        attn_mask = torch.tensor(attn_mask).unsqueeze(0).to(self.device)
        seg_ids = torch.tensor(seg_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            bert_output = self.bert_model(token_ids,
                                          attention_mask=attn_mask,
                                          token_type_ids=seg_ids).last_hidden_state

        return bert_output.squeeze(0), attn_mask.squeeze(0)


def get_data(data_path, config):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_iter = DataIterater(data['train'], config)
    val_iter = DataIterater(data['valid'], config)
    return train_iter, val_iter
