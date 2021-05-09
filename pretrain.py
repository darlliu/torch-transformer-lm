from model import TransformerModel
from tokenizers import BertWordPieceTokenizer
import torch
import math
import csv
import pickle
import os
import time
import numpy as np
import tqdm
from dataclasses import dataclass

SEQ_SIZE = 10
BATCH_SIZE = 50
VOCAB_SIZE = 20000
VAL_SPLIT = 0.15
EPOCHS = 20
EMB_SIZE = 64
TRANSFORMER_DIM = 100
TRANSFORMER_HIDDEN_LAYERS = 2
TRANSFORMER_HEADS = 2
DROPOUT_RATE = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class qac:
    query: str
    vertical: int
    locale: str
    tags: str
    clicks: int
    members: int
    datestr: str


class qac_dataset(object):
    def __init__(self, fname):
        super().__init__()
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        self.data = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():
                line = line.replace('\0', '').strip()
                if "\"" not in line:
                    row = line.split(",")
                else:
                    reader = csv.reader([line], delimiter=',', quotechar='\"')
                    row = next(reader)
                if len(row) != 7:
                    print("incorrect row: {}".format(row))
                    continue
                self.data.append(
                    qac(row[0], int(row[1]), row[2], row[3], int(row[4]), int(row[5]), row[6]))
        return

    def __iter__(self):
        for row in self.data:
            yield row

    def __getitem__(self, idx):
        if idx > len(self.data):
            raise IndexError(
                "{} is too large for dataset size: {}".format(idx, len(self.data)))
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# 1, pre-load the data
if os.path.exists("D:/data/lmdata/positive.pkl") and os.path.exists("D:/data/lmdata/negative.pkl"):
    df_pos = pickle.load(open("D:/data/lmdata/positive.pkl", "rb"))
    df_neg = pickle.load(open("D:/data/lmdata/negative.pkl", "rb"))
else:
    df_pos = qac_dataset("D:/data/lmdata/positive.csv")
    df_neg = qac_dataset("D:/data/lmdata/negative.csv")
    pickle.dump(df_pos, open("D:/data/lmdata/positive.pkl", "wb"))
    pickle.dump(df_neg, open("D:/data/lmdata/negative.pkl", "wb"))

cnt = 0
for row in df_pos:
    print(row)
    cnt += 1
    if cnt > 10:
        break


# 2, tokenize the data using bpe
if os.path.exists("D:/data/lmdata/tokenizer.pkl"):
    tk = pickle.load(open("D:/data/lmdata/tokenizer.pkl", "rb"))
else:
    with open("D:/data/lmdata/vocab_training.txt", "w", encoding="utf-8") as f:
        for q in df_pos:
            f.write(q.query)
            f.write("\n")
        for q in df_neg:
            f.write(q.query)
            f.write("\n")

    tk = BertWordPieceTokenizer()
    tk.train("D:/data/lmdata/vocab_training.txt", vocab_size=VOCAB_SIZE)
    cnt = 0
    for q in df_pos:
        print(q)
        print(" -> {}".format(tk.encode(q.query).tokens))
        cnt += 1
        if cnt > 10:
            break
    pickle.dump(tk, open("D:/data/lmdata/tokenizer.pkl", "wb"))


def data_process(data):
    vals = []
    for q in tqdm.tqdm(data):
        tokens = tk.encode(q.query).tokens
        ids = [tk.token_to_id(t) for t in tokens]
        tensor = torch.tensor(ids, dtype=torch.long)
        vals.append(tensor)
    return vals


print("preloading tensors")
pos_data = data_process(df_pos)
neg_data = data_process(df_neg)


class QACPretrainData(torch.utils.data.Dataset):
    def __init__(self, data: list, label: int):
        self.label = label
        self.data = data
        self.header = torch.tensor([tk.token_to_id("[CLS]")])
        self.sep = torch.tensor([tk.token_to_id("[SEP]")])

    def __getitem__(self, index):
        if index > len(self.data):
            raise IndexError(
                "index too long {} vs {}".format(index, len(self.data)))
        return (self.pad_tensor(self.data[index]).to(device), self.pad_tensor(self.data[index][1:]).to(device))

    def pad_tensor(self, tokens):
        tokens = torch.cat([self.header, tokens, self.sep])
        if len(tokens) > SEQ_SIZE:
            tokens = tokens[:SEQ_SIZE]
        elif len(tokens) < SEQ_SIZE:
            tokens = torch.cat(
                [tokens] + [torch.tensor([tk.token_to_id("[PAD]") for i in range(SEQ_SIZE - len(tokens))])])
        return tokens

    def __len__(self):
        return len(self.data)


ds_pos = QACPretrainData(pos_data, 1)
ds_neg = QACPretrainData(neg_data, 0)


# 3, split into train validate and test set

print("concating dataset")
concat_dataset = torch.utils.data.ConcatDataset([ds_pos, ds_neg])
indices = list(range(len(concat_dataset)))
split = int(np.floor(VAL_SPLIT * len(concat_dataset)))
np.random.shuffle(indices)
val_idxs, train_idxs = indices[:split], indices[split:]

print("constructing sampler")
ds_train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
ds_val_sampler = torch.utils.data.SubsetRandomSampler(val_idxs)

print("finalizing datasets")
ds_train = torch.utils.data.DataLoader(
    concat_dataset, batch_size=BATCH_SIZE, sampler=ds_train_sampler)
ds_val = torch.utils.data.DataLoader(
    concat_dataset, batch_size=BATCH_SIZE, sampler=ds_val_sampler)

# 4, load the model file
print("loading model")
if os.path.exists("D:/data/lmdata/pretrained.model.pkl"):
    model = torch.load("D:/data/lmdata/pretrained.model.pkl")
else:
    model = TransformerModel(VOCAB_SIZE, EMB_SIZE, TRANSFORMER_HEADS,
                             TRANSFORMER_DIM, TRANSFORMER_HIDDEN_LAYERS, DROPOUT_RATE).to(device)
print(model)
# 5, train the model

print("setting up for training")
criterion = torch.nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

print("training")


def train(ds):
    model.train()
    total_loss = 0
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(BATCH_SIZE).to(device)
    for i, batch in enumerate(ds):
        data, targets = batch
        if data.size(0) != BATCH_SIZE:
            continue
        optimizer.zero_grad()
        output = model(data, src_mask)
        loss = criterion(output.view(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(' epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      0, i, len(ds_train), scheduler.get_last_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return


# 6, evaluate the model

def evaluate(ds):
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(BATCH_SIZE).to(device)
    with torch.no_grad():
        for batch in ds:
            data, targets = batch
            if data.size(0) != BATCH_SIZE:
                continue
            output = model(data, src_mask)
            output_flat = output.view(-1, VOCAB_SIZE)
            total_loss += len(data) * criterion(output_flat,
                                                targets.reshape(-1)).item()
    return total_loss / (len(ds) - 1)


best_val_loss = float("inf")
best_model = None

# 7, start pretraining

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(ds_train)
    val_loss = evaluate(ds_val)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(best_model, open(
            "D:/data/lmdata/pretrained.model.pkl", "wb"))

    scheduler.step()
