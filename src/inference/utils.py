import torch


def load_weights(model, PATH):
    if PATH is not None:
        model.load_state_dict(torch.load(PATH))
    model.eval()
    return model