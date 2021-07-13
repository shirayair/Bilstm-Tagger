STUDENT = {'name': "Osnat Ackerman_Shira Yair",
    'ID': '315747204_315389759'}
import torch


def calc_accuracy(predictions, labels, labels_to_idx):
    dom = len(labels)
    good = 0
    hi = 0
    for p, l in zip(predictions, labels):
        if l == labels_to_idx['PAD']:
            hi += 1
            continue
        if p.argmax() == l:
            if not get_key(int(l), labels_to_idx) == 'O':
                good += 1
            else:
               dom -= 1
    if dom == 0:
        return 0
    return good / float(dom - hi)


def get_key(val, labels_to_idx):
    for key, value in labels_to_idx.items():
        if val == value:
            return key
    return None


def validation_check(i, model, valid_loader, loss_func, labels_to_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    # print(f'Epoch: {i + 1:02} | Starting Evaluation...')
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        prediction, loss, y = apply(model, loss_func, x, y)
        epoch_acc += calc_accuracy(prediction, y, labels_to_idx)
        epoch_loss += loss.item()
    # print(f'Epoch: {i + 1:02} | Finished Evaluation')
    return float(epoch_loss) / len(valid_loader), float(epoch_acc) / len(valid_loader)


def apply(model, loss_function, batch, targets):
    predictions, targets = model(batch, targets)
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1)
    loss_ = loss_function(predictions, targets)
    return predictions, loss_, targets


def train_and_eval_model(model, data_loader, valid_loader, loss_func, labels_to_idx, ephocs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(ephocs):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        print(f'Epoch: {i + 1:02} | Starting Training...')
        for (x, y) in data_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predictions, loss, y = apply(model, loss_func, x, y)
            epoch_acc += calc_accuracy(predictions, y, labels_to_idx)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {i + 1:02} | Finished Training')
        avg_epoch_loss, avg_epoch_acc = float(epoch_loss) / len(data_loader), float(epoch_acc) / len(data_loader)
        avg_epoch_loss_val, avg_epoch_acc_val = validation_check(i, model, valid_loader, loss_func, labels_to_idx)
        print(f'\tTrain Loss: {avg_epoch_loss:.3f} | Train Acc: {avg_epoch_acc * 100:.2f}%')
        print(f'\t Val. Loss: {avg_epoch_loss_val:.3f} |  Val. Acc: {avg_epoch_acc_val * 100:.2f}%')
    return model


def train_and_eval_model_500(model, data_loader, valid_loader, loss_func, labels_to_idx, ephocs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracies = []
    sentences_seen = 0
    for i in range(ephocs):
        if sentences_seen % 500 != 0:
            sentences_seen += (500 - (sentences_seen % 500))
        epoch_loss = 0
        epoch_acc = 0
        print(f'Epoch: {i + 1:02} | Starting Training...')
        for (x, y) in data_loader:
            model.train()
            optimizer.zero_grad()
            predictions, loss, y = apply(model, loss_func, x, y)
            epoch_acc += calc_accuracy(predictions, y, labels_to_idx)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            sentences_seen += len(x)
            if sentences_seen % 500 == 0:
                avg_epoch_loss_val, avg_epoch_acc_val = validation_check(i, model, valid_loader, loss_func,
                                                                         labels_to_idx)
                accuracies.append(avg_epoch_acc_val)
                print(f'num of sentences seen: {sentences_seen} | accuracy on dev: {avg_epoch_acc_val * 100:.2f}')

        print(f'Epoch: {i + 1:02} | Finished Training')
        # avg_epoch_loss, avg_epoch_acc = float(epoch_loss) / len(data_loader), float(epoch_acc) / len(data_loader)
        # print(f'\tTrain Loss: {avg_epoch_loss:.3f} | Train Acc: {avg_epoch_acc * 100:.2f}%')
        # print(f'\t Val. Loss: {avg_epoch_loss_val:.3f} |  Val. Acc: {avg_epoch_acc_val * 100:.2f}%')
    return model, accuracies
