import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.nn import MSELoss
from _model import Transformer, MaskEncoderTransformer
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, input1, input2, output):
        self.input1 = input1
        self.input2 = input2
        self.output = output

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        return self.input1[idx], self.input2[idx], self.output[idx]
    



def main():

    BATCH_SIZE = 32
    EPOCHS = 10000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 3e-4
    max_len = 2048
    train_input1 = np.load('E:/Python/Unfinished/dcnn_transformer/input1.npy', allow_pickle=True)
    train_input2 = np.load('E:/Python/Unfinished/dcnn_transformer/input2.npy', allow_pickle=True)
    train_output = np.load('E:/Python/Unfinished/dcnn_transformer/output.npy', allow_pickle=True)

    # 创建数据集实例
    train_dataset = TrajectoryDataset(train_input1, train_input2, train_output)

    train_loader =  DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"--------data loaded--------")
    try:
        src_ais_size = len(train_input1[0, 0])
        trg_ais_size = len(train_output[0, 0])
    except:
        src_ais_size = 1
        trg_ais_size = 1
    embedding_size = 512
    num_heads = 8
    num_layers = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    forward_expansion = 4
    src_pad_idx = 0
    trg_pad_idx = 0

    model = MaskEncoderTransformer(
        src_ais_size,
        trg_ais_size,
        src_pad_idx,
        trg_pad_idx,
        embedding_size,
        num_layers,
        forward_expansion,
        num_heads,
        dropout,
        DEVICE,
        max_len
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
            model_output = model_output.reshape(-1, model_output.shape[2])
            output = output[:, 1:].reshape(-1)
            model_output = model_output.to(torch.float32)
            output = output.to(torch.float32)

            loss = crition(output, model_output) * 0.01

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