from .vzsl import VZSL


def create_model(opt):
    if opt.model == 'vzsl':
        model = VZSL(opt)

    else:
        raise Exception()

    return model
