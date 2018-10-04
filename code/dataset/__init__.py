from .awa import AwA


def create_dataset(opt, split):
    if opt.dataset == 'awa':
        dataset = AwA(opt, split)
    else:
        raise Exception()

    assert opt.vdim == dataset.vdim
    assert opt.sdim == dataset.sdim
    assert opt.n_classes == dataset.n_classes

    return dataset
