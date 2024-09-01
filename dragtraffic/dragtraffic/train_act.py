import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dragtraffic.act.model.tg_act import actuator
from dragtraffic.act.model.tg_act_diff import actuatordiff
from dragtraffic.act.model.tg_act_drag import actuatordrag
from dragtraffic.act.model.tg_act_led import actuatorled
from dragtraffic.act.utils.act_dataset import actDataset
from dragtraffic.utils.config import get_parsed_args
from dragtraffic.utils.config import load_config_act

import torch.utils.data as data


if __name__ == '__main__':
    args = get_parsed_args()
    cfg = load_config_act(args.config)

    if cfg['init_model_config'] is not None:
        init_cfg = load_config_act(cfg['init_model_config'])
        cfg['init_model_config'] = init_cfg

    if cfg['diff_model_config'] is not None:
        diff_cfg = load_config_act(cfg['diff_model_config'])
        cfg['diff_model_config'] = diff_cfg

    cfg['use_cache'] = args.cache
    cfg['eval_interval'] = args.eval_interval
    seed_everything(2024, workers=True)
    torch.set_float32_matmul_precision('medium')

    ckpt_callback = ModelCheckpoint(
        monitor='val/pos_loss' if cfg['model'] != 'diffusion' else 'val/diffusion_loss',
        save_top_k=3,
        mode='min',
        verbose=True,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        auto_insert_metric_name=True,
    )

    last_callback = ModelCheckpoint(
        every_n_epochs=cfg['max_epoch'],
        save_top_k=-1,
        verbose=True,
        filename='last-{}'.format(args.exp_name),
        auto_insert_metric_name=True
    )

    if cfg['debug']:
        trainer = pl.Trainer(devices=1, gradient_clip_val=0.5, accelerator=cfg['device'], profiler="simple")
    else:
        trainer = pl.Trainer(
            check_val_every_n_epoch=cfg['eval_interval'],
            max_epochs=cfg['max_epoch'],
            logger=TensorBoardLogger("lightning_logs", name=args.exp_name),
            devices=args.devices,
            gradient_clip_val=0.5,
            accelerator=cfg['device'],
            profiler="simple",
            strategy=cfg['strategy'],
            val_check_interval=cfg['eval_interval'],
            callbacks=[ckpt_callback, last_callback],
        )

    train_set = actDataset(cfg)

    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = int(len(train_set) * 0.1)
    test_set_size = len(train_set) - train_set_size - valid_set_size

    seed = torch.Generator().manual_seed(2023)
    train_set, val_set, test_set = data.random_split(train_set, [train_set_size, valid_set_size, test_set_size], generator=seed)

    train_loader = DataLoader(
        train_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False, drop_last=False
    )

    test_loader = DataLoader(
        test_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False, drop_last=False
    )

    if args.eval:
        print('evaluating model...')
        if cfg['model'] == 'diffusion':
            model = actuatordiff.load_from_checkpoint(args.model_path)
        elif cfg['model'] == 'drag':
            model = actuatordrag.load_from_checkpoint(args.model_path)
        elif cfg['model'] == 'led':
            model = actuatorled.load_from_checkpoint(args.model_path)
        else:
            model = actuator.load_from_checkpoint(args.model_path)
        trainer.validate(model, test_loader)
    else:
        print('training model...')
        if cfg['model'] == 'diffusion':
            model = actuatordiff(cfg)
        elif cfg['model'] == 'drag':
            model = actuatordrag(cfg)
        elif cfg['model'] == 'led':
            model = actuatorled(cfg)
        else:
            model = actuator(cfg)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.model_path if args.model_path != '' else None)
