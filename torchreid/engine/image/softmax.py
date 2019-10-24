from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss, AMSoftmaxLoss, AdaCosLoss, DSoftmaxLoss, MetricLosses
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, scheduler=None, use_gpu=False,
                 softmax_type='stock', label_smooth=True, conf_penalty=False,
                 m=0.35, s=10):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, reg_cfg, optimizer, scheduler, use_gpu)

        if softmax_type == 'stock':
            self.criterion = CrossEntropyLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                conf_penalty=conf_penalty
            )
        elif softmax_type == 'am':
            self.criterion = AMSoftmaxLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                conf_penalty=conf_penalty,
                m=m, s=s
            )
        elif softmax_type == 'ada':
            self.criterion = AdaCosLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                conf_penalty=conf_penalty
            )
        elif softmax_type == 'd_sm':
            self.criterion = DSoftmaxLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                conf_penalty=conf_penalty
            )

        if metric_cfg.enabled:
            self.metric_losses = MetricLosses(self.datamanager.num_train_pids,
                                              256, self.writer, metric_cfg.soft_margin)
            self.metric_losses.center_coeff = metric_cfg.center_coeff
            self.metric_losses.glob_push_plus_loss_coeff = metric_cfg.glob_push_plus_loss_coeff
            self.metric_losses.push_loss_coeff = metric_cfg.push_loss_coeff
            self.metric_losses.push_plus_loss_coeff = metric_cfg.push_plus_loss_coeff
            self.metric_losses.min_margin_loss_coeff = metric_cfg.min_margin_loss_coeff
        else:
            self.metric_losses = None

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        reg_ow_loss = AverageMeter()
        reg_of_loss = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            self.optimizer.zero_grad()
            if self.of_regularizer:
                outputs, feature_maps = self.model(imgs, get_of_outputs=True)
            elif self.metric_losses is not None:
                outputs, embeddings = self.model(imgs, get_embedding=True)
            else:
                outputs = self.model(imgs)
                feature_maps = []
            loss = self._compute_loss(self.criterion, outputs, pids)

            if (epoch+1) > fixbase_epoch:
                if self.of_regularizer:
                    of_reg_loss = self.of_regularizer(feature_maps)
                    reg_of_loss.update(of_reg_loss.item(), pids.size(0))
                else:
                    of_reg_loss = 0
                reg_loss = self.regularizer(self.model)
                reg_ow_loss.update(reg_loss.item(), pids.size(0))
                loss += reg_loss + of_reg_loss

            if self.metric_losses is not None:
                self.metric_losses.writer = self.writer
                self.metric_losses.init_iteration()
                metric_loss, info = self.metric_losses(embeddings, pids, epoch, epoch * num_batches + batch_idx)
                self.metric_losses.end_iteration()
                loss += metric_loss

            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accs,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )

                if self.writer is not None:
                    n_iter = epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                    self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                    info = self.criterion.get_last_info()
                    for k in info:
                        self.writer.add_scalar('AUX info/' + k, info[k], n_iter)
                    self.writer.add_scalar('Loss/train', losses.avg, n_iter)
                    self.writer.add_scalar('Loss/reg_ow', reg_ow_loss.avg, n_iter)
                    self.writer.add_scalar('Loss/reg_of', reg_of_loss.avg, n_iter)
                    self.writer.add_scalar('Accuracy/train', accs.avg, n_iter)
                    self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], n_iter)

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
