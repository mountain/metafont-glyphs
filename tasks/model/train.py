import argparse
import torch as th
import lightning.pytorch as pl

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

if th.cuda.is_available():
    accelerator = 'gpu'
    th.set_float32_matmul_precision('medium')
elif th.backends.mps.is_available():
    accelerator = 'cpu'
else:
    accelerator = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("-d", "--direction", type=str, default='g2m', help="model direction")
parser.add_argument("-m", "--model", type=str, default='bs', help="model to execute")
opt = parser.parse_args()

# data
if __name__ == '__main__':
    # model
    import importlib
    mdl = importlib.import_module('%s.%s' % (opt.direction, opt.model), package=None)
    model = mdl._model_()

    print('loading data...')

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])

    print('training...')
    trainer.fit(model)

    print('testing...')
    trainer.test(ckpt_path="best")
