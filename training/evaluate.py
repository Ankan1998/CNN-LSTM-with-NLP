import torch


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            output = model(batch.text)
            loss = criterion(output, torch.unsqueeze(batch.labels,1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)