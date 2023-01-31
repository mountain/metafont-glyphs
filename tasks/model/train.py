import argparse
import pytorch_lightning as pl

from data.dataset import ValidDataset, TestDataset, TrainDataset

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("-d", "--direction", type=str, default='m2g', help="model direction")
parser.add_argument("-m", "--model", type=str, default='v0', help="model to execute")
opt = parser.parse_args()

print('loading data...')
import data.dataset as ds

# data
ds_train, ds_val, ds_test = TrainDataset(), ValidDataset(), TestDataset()

if __name__ == '__main__':
    # model
    import importlib
    mdl = importlib.import_module('%s.%s' % (opt.direction, opt.model), package=None)
    model = mdl._model_()

    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=32,
                         max_epochs=opt.n_epochs, reload_dataloaders_every_epoch=True)
    trainer.fit(model)
    trainer.test(ckpt_path="best")
