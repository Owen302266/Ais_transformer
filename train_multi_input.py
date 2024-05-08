import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.nn import MSELoss
from _model import MaskEncoderTransformer
import numpy as np
from _model_multi_input import Transformer
import glob
import os

    
class AisDataset(Dataset):
    def __init__(self, path, transform=None):
        self.encoder_input = []
        self.decoder_input = []
        self.model_output = []
        self.transform = transform
        self.load_data2(path)

    def __len__(self):
        return len(self.encoder_input)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # tensor to list
            idx = idx.tolist()
        return self.encoder_input[idx], self.decoder_input[idx], self.model_output[idx]
    
    def load_data(self, path):
        # input 3 npy
        for mode in ['encoder_input', 'decoder_input', 'model_output']:
            npy_list = glob.glob(os.path.join(path, mode) + '/*.npy')
            for npy_file in npy_list:
                ais_data = np.load(npy_file, allow_pickle=True)
                for data in ais_data:
                    if mode == 'encoder_input':
                        self.encoder_input.append(data)
                    elif mode == 'decoder_input':
                        self.decoder_input.append(data)
                    else:
                        self.model_output.append(data)

    def load_data2(self, path):
        # input single npy
        npy_list = glob.glob(path + '/*.npy')
        for npy_file in npy_list:
            ais_data = np.load(npy_file, allow_pickle=True)
            for data in ais_data:
                self.encoder_input.append(data[:-1])
                self.decoder_input.append(data[1:])
                for i in range(len(data) - 1):
                    data[i, 0] = data[i + 1, 0]
                self.model_output.append(data[:-1])


def main():

    BATCH_SIZE = 32
    EPOCHS = 10000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 3e-4
    max_len = 2048

    # 创建数据集实例
    train_dataset = AisDataset('E:/Python/Unfinished/Ais_transformer/dataset')

    train_loader =  DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"--------data loaded--------")

    embedding_size = 512
    num_heads = 8
    num_layers = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    forward_expansion = 4
    src_pad_idx = 0
    trg_pad_idx = 0

    model = Transformer(
        d_model=128,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=num_encoder_layers,
        dropout=dropout,
        batch_first=True,
    ).to(DEVICE)

    crition = MSELoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    mode = 'train'

    for epoch in range(EPOCHS):
        print(f"[Epoch {epoch} / {EPOCHS}]")
        losses = []

        for idx, (input1, input2, output) in enumerate(train_loader):
            input1 = input1.to(DEVICE)
            input2 = input2.to(DEVICE)
            output = output.to(DEVICE)

            model_output = model(input1, input2)

            optimizer.zero_grad()
            # model_output = model_output.reshape(-1, model_output.shape[2])
            # output = output[:, 1:].reshape(-1)
            model_output = model_output.to(torch.float32)
            output = output.to(torch.float32)

            loss = crition(output, model_output)

            losses.append(loss.item())
            loss = loss.float()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            if idx % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch, EPOCHS, idx//100, loss.item()))

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)


if __name__ == '__main__':
    main()