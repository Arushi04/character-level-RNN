import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

from dataset import fetch_dataset
from dataset import Vocab
from model import RNN

def main(args):
    print("in main")
    #creating tensorboard object
    tb_writer = SummaryWriter(log_dir=os.path.join(args.outdir, "tb/"), purge_step=0)

    #Loading data
    train_dl, val_dl, vocab, label_map = fetch_dataset(args.datapath)

    #Defining loss
    criterion = nn.CrossEntropyLoss()

    #Defining optimizer
    vocab_size = len(vocab)
    num_classes = len(label_map)
    model = RNN(vocab_size, num_classes, args.embed_dim, args.hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Looping training data
    for epoch in range(args.epochlen):
        running_loss, test_loss = 0.0, 0.0
        count = 0
        correct = 0
        total_labels = 0
        all_train_loss = []
        all_test_loss = []
        model.train()
        best_accuracy = 0
        for i, batch in enumerate(train_dl):
            seqs, labels = batch

            #names = Vocab.get_string(batch)

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
        all_train_loss.append(total_loss)
        accuracy = (correct * 100) / total_labels
        tb_writer.add_scalar('Train_Loss', running_loss, epoch)
        tb_writer.add_scalar('Train_Accuracy', accuracy, epoch)

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
        all_test_loss.append(total_test_loss)
        test_accuracy = (correct * 100) / total_labels
        print(f"Epoch : {str(epoch).zfill(2)}, Training Loss : {round(total_loss, 4)}, Training Accuracy : {round(accuracy, 4)},"
              f" Test Loss : {round(total_test_loss, 4)}, Test Accuracy : {round(test_accuracy, 4)}")
        tb_writer.add_scalar('Test_Loss', test_loss, epoch)
        tb_writer.add_scalar('Test_Accuracy', test_accuracy, epoch)

        if best_accuracy < test_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), args.outdir + args.modelname + str(epoch))

    # Plot confusion matrix
    y_true = []
    y_pred = []
    for data in val_dl:
        seq, labels = data
        outputs = model(seq)
        predicted = torch.argmax(outputs, dim=1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.keys())
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--datapath", type=str, default="data/names/", help="")
    parser.add_argument("--outdir", type=str, default="./output/", help="")
    parser.add_argument("--modelname", type=str, default="modelv", help="")
    parser.add_argument("--epochlen", type=int, default=13, help="")
    parser.add_argument("--lr", type=int, default=.005, help="")
    parser.add_argument("--embed_dim", type=int, default=50, help="")
    parser.add_argument("--hidden_size", type=int, default=100, help="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)