import cfg
from src.core.yaml_utils import create
import torch


class YAMLConfig():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = cfg.update_cfg(args.yaml_path, vars(args))

        # model
        self._model = None
        self._criterion = None
        # Dataset
        self._train_dataloader = None
        self._val_dataloader = None
        self._postprocessor = None
        # Optimizer
        self._lr_scheduler = None
        self._optimizer = None
        self._scaler = None
        self._ema = None


        self.output_dir = self.config.get('output_dir', 'output')

        # runtime
        self.resume = self.config.get('resume', None)
        self.tuning = self.config.get('tuning', None)
        self.epoches = self.config.get('epoches', 100)
        self.last_epoch = self.config.get('last_epoch', 0)
        self.end_epoch = self.config.get('end_epoch', None)

        # setting
        self.checkpoint_step = self.config.get('checkpoint_step', 1)
        self.log_step = self.config.get('epoches', 100)

        # training
        self.use_amp = self.config.get('use_amp', True)
        self.use_ema = self.config.get('use_ema', True)
        self.sync_bn = self.config.get('sync_bn', True)
        self.clip_max_norm = self.config.get('clip_max_norm', 0.1)
        self.find_unused_parameters = self.config.get('find_unused_parameters', None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    @property
    def model(self):
        if self._model is None:
            self._model = create(self.config['model'])
        return self._model

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = create(self.config['criterion'])
        return self._criterion

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = create(self.config['optimizer'], params=self.model.parameters())
        return self._optimizer

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.config:
            self._postprocessor = create(self.config['postprocessor'])
        return self._postprocessor

    @property
    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = create(self.config['train_dataloader'])
        return self._train_dataloader

    @property
    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = create(self.config['val_dataloader'])
        return self._val_dataloader


    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._lr_scheduler = create(self.config['lr_scheduler'], optimizer=self.optimizer)
        return self._lr_scheduler


    @property
    def ema(self, ):
        if self._ema is None and self.config.get('use_ema', False):
            self._ema = create(self.config['ema'], model=self.model)
        return self._ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.config.get('use_amp', False):
            self._scaler = create({'type': 'GradScaler'})
        return self._scaler
