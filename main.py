import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import fetch_dataset
from model import RNN

def main(args):
    print("in main")
    #creating tensorboard object
    writer = SummaryWriter(log_dir=os.path.join(args.outdir, "train/"), purge_step=0)

    #Loading data
    train_dl, val_dl, vocab, label_map = fetch_dataset(args.datapath)

    #Defining loss
    criterion = nn.CrossEntropyLoss()

    #Defining optimizer
    vocab_size = len(vocab)
    num_classes = len(label_map)
    model = RNN(vocab_size, num_classes, args.embed_dim, args.hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    #Looping training data
    for epoch in range(args.epochlen):
        running_loss, test_loss = 0.0, 0.0
        count = 0
        correct = 0
        total_labels = 0

        model.train()
        for i, batch in enumerate(train_dl):
            seqs, labels = batch

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            pred_outputs = model(seqs)
            loss = criterion(pred_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1

            correct += (torch.argmax(pred_outputs, dim=1) == labels).sum().item()
            total_labels += labels.size(0)
        total_loss = running_loss/count
        accuracy = (correct * 100) / total_labels

        count = 0
        model.eval()
        for batch in val_dl:
            seqs, labels = batch

            pred_outputs = model(seqs)
            loss = criterion(pred_outputs, labels)
            test_loss += loss.item()
            count += 1

            correct += (torch.argmax(pred_outputs, dim=1) == labels).sum().item()
            total_labels += labels.size(0)
        total_test_loss = test_loss/count
        test_accuracy = (correct * 100) / total_labels
        print(f"Epoch : {str(epoch).zfill(2)}, Training Loss : {round(total_loss, 4)}, Training Accuracy : {round(accuracy, 4)},"
              f" Test Loss : {round(total_test_loss, 4)}, Test Accuracy : {round(test_accuracy, 4)}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--datapath", type=str, default="data/names/", help="")
    parser.add_argument("--outdir", type=str, default="./output/", help="")
    parser.add_argument("--epochlen", type=int, default=10, help="")
    parser.add_argument("--embed_dim", type=int, default=30, help="")
    parser.add_argument("--hidden_size", type=int, default=80, help="")
    args = parser.parse_args()
    main(args)