from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import os.path as osp
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.losses import DeepSupervision
from torchreid.utils import (AverageMeter, MetricMeter, get_model_attr,
                             open_all_layers, open_specified_layers,
                             re_ranking, save_checkpoint,
                             visualize_ranked_results)


class Engine:
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, models, optimizers, schedulers,
                 use_gpu=True, save_chkpt=True, train_patience = 10, lb_lr = 1e-5, early_stoping=False):

        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.save_chkpt = save_chkpt
        self.writer = None

        self.start_epoch = 0
        self.fixbase_epoch = 0
        self.iter_to_wait = 0
        self.best_metric = 0.0
        self.max_epoch = None
        self.num_batches = None
        self.epoch = None
        self.lb_lr = lb_lr
        self.train_patience = train_patience
        self.early_stoping = early_stoping

        self.models = OrderedDict()
        self.optims = OrderedDict()
        self.scheds = OrderedDict()

        if isinstance(models, (tuple, list)):
            assert isinstance(optimizers, (tuple, list))
            assert isinstance(schedulers, (tuple, list))

            num_models = len(models)
            assert len(optimizers) == num_models
            assert len(schedulers) == num_models

            for model_id, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
                self.register_model(f'model_{model_id}', model, optimizer, scheduler)
        else:
            assert not isinstance(optimizers, (tuple, list))
            assert not isinstance(schedulers, (tuple, list))

            self.register_model('model', models, optimizers, schedulers)

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self.models[name] = model
        self.optims[name] = optim
        self.scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self.models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, save_dir, is_best=False):
        names = self.get_model_names()
        main_model_name = names[0]

        for name in names:
            ckpt_path = save_checkpoint(
                            {
                                'state_dict': self.models[name].state_dict(),
                                'epoch': epoch + 1,
                                'optimizer': self.optims[name].state_dict(),
                                'scheduler': self.scheds[name].state_dict(),
                                'num_classes': self.datamanager.num_train_pids,
                                'classes_map': self.datamanager.train_loader.dataset.classes
                            },
                            osp.join(save_dir, name),
                            is_best=is_best
                        )

            if name == main_model_name:
                latest_name = osp.join(save_dir, 'latest.pth')
                if osp.lexists(latest_name):
                    os.remove(latest_name)
                os.symlink(ckpt_path, latest_name)

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self.models[name].train()
            else:
                self.models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self.optims[name].param_groups[0]['lr']

    def update_lr(self, names=None, output_avg_metric=None):
        names = self.get_model_names(names)

        for name in names:
            if self.scheds[name] is not None:
                if self.scheds[name].__class__.__name__ in ['ReduceLROnPlateau', 'WarmupScheduler']:
                    self.scheds[name].step(metrics=output_avg_metric)
                else:
                    self.scheds[name].step()

    def exit_on_plataeu(self, top1, top5, mAP):
            name = self.get_model_names()[0]
            current_lr = self.get_current_lr()

            if current_lr == self.lb_lr:
                current_metric = np.round(top1, 4)
                if (self.best_metric >= current_metric):
                    self.iter_to_wait += 1
                    if self.iter_to_wait >= self.train_patience:
                        print("The training stopped due to no improvements for {} epochs".format(self.train_patience))
                        return True
                else:
                    self.best_metric = current_metric
                    self.iter_to_wait = 0

            return False

    def run(
        self,
        save_dir='log',
        tb_log_dir='',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        lr_finder = False,
        **kwargs
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """
        if visrank and not test_only:
            raise ValueError('visrank can be set to True only if test_only=True')

        if test_only:
            for name in self.get_model_names():
                opti = self.optims[name]
                sch = self.scheds[name]
                lr = opti.param_groups[0]['lr']
                print(f'lr: {name}, {lr}')
                print(f'sch: {name}, {sch}')
                print(f'epoch: {start_epoch}')

            self.test(
                0,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
            )
            return

        if self.writer is None:
            log_dir = tb_log_dir if len(tb_log_dir) else save_dir
            self.writer = SummaryWriter(log_dir=log_dir)

        time_start = time.time()
        top1 = None
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.fixbase_epoch = fixbase_epoch
        print('=> Start training')

        for self.epoch in range(self.start_epoch, self.max_epoch):

            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers,
                lr_finder = lr_finder
            )

            if ((self.epoch + 1) >= start_eval
               and eval_freq > 0
               and (self.epoch+1) % eval_freq == 0
               and (self.epoch + 1) != self.max_epoch):

                top1, top5, mAP = self.test(
                    self.epoch,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    lr_finder = lr_finder
                )
                if self.save_chkpt and not lr_finder:
                    self.save_model(self.epoch, save_dir)

                if lr_finder:
                    print(f"epoch: {self.epoch}\t top1: {top1}\t lr: {self.get_current_lr()}")

            if (self.early_stoping and
                not lr_finder and
                self.exit_on_plataeu(top1, top5, mAP)):
                    break

        if self.max_epoch > 0:
            print('=> Final test')
            top1_final, top5, mAP = self.test(
                self.epoch,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                lr_finder = lr_finder
            )
            if self.save_chkpt and not lr_finder:
                self.save_model(self.epoch, save_dir)
            if lr_finder:
                print(f"epoch: {self.epoch}\t top1: {top1_final}\t lr: {self.get_current_lr()}")

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))

        if self.writer is not None:
            self.writer.close()

        return top1_final

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None, lr_finder=False):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        accuracy = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            loss_summary, avg_acc = self.forward_backward(data)
            batch_time.update(time.time() - end)

            losses.update(loss_summary)
            accuracy.update(avg_acc)

            if not lr_finder and ((self.batch_idx + 1) % print_freq) == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (self.epoch + 1)) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'cls acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        accuracy=accuracy,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            if self.writer is not None and not lr_finder:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                self.writer.add_scalar('Aux/lr', self.get_current_lr(), n_iter)
                self.writer.add_scalar('Accuracy/train', accuracy.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Loss/' + name, meter.avg, n_iter)

            end = time.time()

        if not lr_finder:
            self.update_lr(output_avg_metric = losses.meters['loss'].avg)

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
        self,
        epoch,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        lr_finder = False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())
        top1, mAP, top5 = [None]*3
        for dataset_name in targets:
            domain = 'source' if dataset_name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(dataset_name, domain))

            for model_name, model in self.models.items():
                if get_model_attr(model, 'classification'):
                    top1, top5, mAP = self._evaluate_classification(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['query'],
                        model_name=model_name,
                        dataset_name=dataset_name,
                        ranks=ranks,
                        lr_finder = lr_finder
                    )
                elif get_model_attr(model, 'contrastive'):
                    pass
                elif dataset_name == 'lfw':
                    self._evaluate_pairwise(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['pairs'],
                        model_name=model_name
                    )
                else:
                    top1, top5, mAP = self._evaluate_reid(
                        model=model,
                        epoch=epoch,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        query_loader=self.test_loader[dataset_name]['query'],
                        gallery_loader=self.test_loader[dataset_name]['gallery'],
                        dist_metric=dist_metric,
                        normalize_feature=normalize_feature,
                        visrank=visrank,
                        visrank_topk=visrank_topk,
                        save_dir=save_dir,
                        use_metric_cuhk03=use_metric_cuhk03,
                        ranks=ranks,
                        rerank=rerank,
                        lr_finder = lr_finder
                    )
        return  top1, top5, mAP

    @torch.no_grad()
    def _evaluate_classification(self, model, epoch, data_loader, model_name, dataset_name, ranks, lr_finder):
        labelmap = []

        if data_loader.dataset.classes and get_model_attr(model, 'classification_classes') and \
                len(data_loader.dataset.classes) < len(get_model_attr(model, 'classification_classes')):

            for class_name in sorted(data_loader.dataset.classes.keys()):
                labelmap.append(data_loader.dataset.classes[class_name])

        cmc, mAP, norm_cm = metrics.evaluate_classification(data_loader, model, self.use_gpu, ranks, labelmap)

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)
            for i, r in enumerate(ranks):
                self.writer.add_scalar('Val/{}/{}/Rank-{}'.format(dataset_name, model_name, r), cmc[i], epoch + 1)

        if not lr_finder:
            print('** Results ({}) **'.format(model_name))
            print('mAP: {:.2%}'.format(mAP))
            for i, r in enumerate(ranks):
                print('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
            if norm_cm.shape[0] <= 20:
                metrics.show_confusion_matrix(norm_cm)

        return cmc[0], cmc[1], mAP

    @torch.no_grad()
    def _evaluate_pairwise(self, model, epoch, data_loader, model_name):
        same_acc, diff_acc, overall_acc, auc, avg_optimal_thresh = metrics.evaluate_lfw(
            data_loader, model, verbose=False
        )

        if self.writer is not None:
            self.writer.add_scalar('Val/LFW/{}/same_accuracy'.format(model_name), same_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/diff_accuracy'.format(model_name), diff_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/accuracy'.format(model_name), overall_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/AUC'.format(model_name), auc, epoch + 1)

        print('\n** Results ({}) **'.format(model_name))
        print('Accuracy: {:.2%}'.format(overall_acc))
        print('Accuracy on positive pairs: {:.2%}'.format(same_acc))
        print('Accuracy on negative pairs: {:.2%}'.format(diff_acc))
        print('Average threshold: {:.2}'.format(avg_optimal_thresh))

    @torch.no_grad()
    def _evaluate_reid(
        self,
        model,
        epoch,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        model_name='',
        lr_finder = False
    ):
        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs),
                features = features.data.cpu()

                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)

            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)

            return f_, pids_, camids_

        qf, q_pids, q_camids = _feature_extraction(query_loader)
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)

        if normalize_feature:
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)
            for r in ranks:
                self.writer.add_scalar('Val/{}/{}/Rank-{}'.format(dataset_name, model_name, r), cmc[r - 1], epoch + 1)
        if not lr_finder:
            print('** Results ({}) **'.format(model_name))
            print('mAP: {:.2%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

        if visrank and not lr_finder:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0], cmc[1], mAP

    @staticmethod
    def compute_loss(criterion, outputs, targets, **kwargs):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets, **kwargs)
        else:
            loss = criterion(outputs, targets, **kwargs)
        return loss

    @staticmethod
    def parse_data_for_train(data, output_dict=False, enable_masks=False, use_gpu=False):
        imgs = data[0]

        obj_ids = data[1]
        if use_gpu:
            imgs = imgs.cuda()
            obj_ids = obj_ids.cuda()

        if output_dict:
            if len(data) > 3:
                dataset_ids = data[3].cuda() if use_gpu else data[3]

                masks = None
                if enable_masks:
                    masks = data[4].cuda() if use_gpu else data[4]

                attr = [record.cuda() if use_gpu else record for record in data[5:]]
                if len(attr) == 0:
                    attr = None
            else:
                dataset_ids = torch.zeros_like(obj_ids)
                masks = None
                attr = None

            return dict(img=imgs, obj_id=obj_ids, dataset_id=dataset_ids, mask=masks, attr=attr)
        else:
            return imgs, obj_ids

    @staticmethod
    def parse_data_for_eval(data):
        imgs = data[0]
        obj_ids = data[1]
        cam_ids = data[2]

        return imgs, obj_ids, cam_ids

    def two_stepped_transfer_learning(self, epoch, fixbase_epoch, open_layers):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))

            for model in self.models.values():
                open_specified_layers(model, open_layers, strict=False)
        else:
            for model in self.models.values():
                open_all_layers(model)
