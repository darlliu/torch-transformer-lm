import torch
import os
import pickle
from model import TransformerModel
from tokenizers import BertWordPieceTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1, run inference on a few given examples for pretrained model
SEQ_SIZE = 10
VOCAB_SIZE = 20000
TEST_QUERIES = [
    "software",
    "amazon software",
    "facebook product",
    "coronavirus",
    "marketing",
    "director of machine",
    "linedin senior",
    "google deep",
    "netease java",
    "alibaba engineering",
    "machine learning",
    "I'm not feeling great"
]

print("loading tokenizer")
if os.path.exists("D:/data/lmdata/tokenizer.pkl"):
    tk = pickle.load(open("D:/data/lmdata/tokenizer.pkl", "rb"))
else:
    raise ImportError("Need tokenizer")

print("loading pretrained model")
if os.path.exists("D:/data/lmdata/pretrained.model.pkl"):
    model = torch.load("D:/data/lmdata/pretrained.model.pkl").to(device)
else:
    raise ImportError("Need pretrained model")


def pad_tensor(tokens):
    header = torch.tensor([tk.token_to_id("[CLS]")])
    sep = torch.tensor([tk.token_to_id("[SEP]")])
    tokens = torch.tensor([tk.token_to_id(token)
                           for token in tk.encode(tokens).tokens])
    tokens = torch.cat([header, tokens, sep])
    if len(tokens) > SEQ_SIZE:
        tokens = tokens[:SEQ_SIZE]
    elif len(tokens) < SEQ_SIZE:
        tokens = torch.cat(
            [tokens] + [torch.tensor([tk.token_to_id("[PAD]") for i in range(SEQ_SIZE - len(tokens))])])
    return tokens


def test_pretrained():
    model.eval()
    src_mask = model.generate_square_subsequent_mask(1).to(device)
    for query in TEST_QUERIES:
        t = pad_tensor(query).reshape((-1, SEQ_SIZE)).to(device)
        print("-"*89)
        output = model.forward(t, src_mask)
        output_idx = torch.squeeze(torch.argmax(output, 2)).tolist()
        print("testing query prediction, {} -> {} (predicted)".format(query,
                                                                      tk.decode(output_idx)))
        print("tensor: {}".format(output))
        print("-"*89)


test_pretrained()
