import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from .lstm import ActionClassificationLSTM, PoseDataModule, WINDOW_SIZE

DATASET_PATH = 'content/RNN-HAR-2D-Pose-database/'

def do_training_validation(window_size=34, batch_size=512, epochs=400, data_root=DATASET_PATH, num_layers=2, learning_rate=0.001, number_of_class=6):
    pl.seed_everything(21)
    hidden_dim = 50
    num_layers = 1
    # Initialize model and data module with provided arguments
    model = ActionClassificationLSTM(window_size, hidden_dim, number_of_class, num_layers, learning_rate=learning_rate)
    data_module = PoseDataModule(data_root=data_root, batch_size=batch_size)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        #gpus=1,
        #progress_bar_refresh_rate=1,
        callbacks=[EarlyStopping(monitor='train_loss', patience=15), checkpoint_callback, lr_monitor]
    )

    # Train the model
    trainer.fit(model, data_module)

    return model

if __name__ == '__main__':
    do_training_validation()
