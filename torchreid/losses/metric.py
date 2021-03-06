import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CenterLoss(nn.Module):
    """Implements the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""
    def __init__(self, num_classes, embed_size, cos_dist=True):
        super().__init__()
        self.cos_dist = cos_dist
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size).cuda())
        self.embed_size = embed_size
        self.mse = nn.MSELoss(reduction='elementwise_mean')

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, features, labels):
        features = F.normalize(features)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size

        if self.cos_dist:
            self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

        centers_batch = self.centers[labels, :]

        if self.cos_dist:
            cos_sim = nn.CosineSimilarity()
            cos_diff = 1. - cos_sim(features, centers_batch)
            center_loss = torch.sum(cos_diff) / batch_size
        else:
            center_loss = self.mse(centers_batch, features)

        return center_loss


class MinimumMargin(nn.Module):
    """Implements the Minimum margin loss from https://arxiv.org/abs/1805.06741"""
    def __init__(self, margin=.6):
        super().__init__()
        self.margin = margin

    def forward(self, centers, labels):
        loss_value = 0

        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])
        valid_pairs = (all_pairs * np.tri(*all_pairs.shape, k=-1, dtype=np.bool)).astype(np.float32)
        losses = 1. - torch.mm(batch_centers, torch.t(batch_centers)) - self.margin

        valid_pairs *= (losses.cpu().data.numpy() > 0.0)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class GlobalPushPlus(nn.Module):
    """Implements the Global Push Plus loss"""
    def __init__(self, margin=.6, soft_margin=False):
        super().__init__()
        self.min_margin = 0.15
        self.max_margin = margin
        self.num_calls = 0
        self.soft_margin = soft_margin

    def forward(self, features, centers, labels):
        self.num_calls += 1
        features = F.normalize(features)
        loss_value = 0
        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        center_ids = np.arange(centers.shape[0], dtype=np.int32)
        different_class_pairs = labels.reshape([-1, 1]) != center_ids.reshape([1, -1])

        pos_distances = 1.0 - torch.sum(features * batch_centers, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(centers))

        margin = self.min_margin + float(self.num_calls) / float(40000) * (self.max_margin - self.min_margin)
        margin = min(margin, self.max_margin)

        if self.soft_margin:
            losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)
        else:
            losses = margin + pos_distances.view(-1, 1) - neg_distances

        valid_pairs = (different_class_pairs * (losses.cpu().data.numpy() > 0.0)).astype(np.float32)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class PushPlusLoss(nn.Module):
    """Implements the Push Plus loss"""
    def __init__(self, margin=.7, soft_margin=False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin

    def forward(self, features, centers, labels):
        features = F.normalize(features)
        loss_value = 0
        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])
        pos_distances = 1.0 - torch.sum(features * batch_centers, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(features))

        if self.soft_margin:
            losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)
        else:
            losses = self.margin + pos_distances.view(-1, 1) - neg_distances
        valid_pairs = (all_pairs * (losses.cpu().data.numpy() > 0.0)).astype(np.float32)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class PushLoss(nn.Module):
    """Implements the Push loss"""
    def __init__(self, margin=0.5, soft_margin=True):
        super().__init__()
        self.soft = soft_margin
        self.margin = margin

    def forward(self, features, labels):
        features = F.normalize(features)
        loss_value = 0
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])
        valid_pairs = (all_pairs * np.tri(*all_pairs.shape, k=-1, dtype=np.bool)).astype(np.float32)

        if self.soft:
            losses = F.softplus(torch.mm(features, torch.t(features)) - 1)
        else:
            losses = self.margin - (1. - torch.mm(features, torch.t(features)))
            valid_pairs *= (losses.cpu().data.numpy() > 0.0)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class MetricLosses:
    """Class-aggregator for all metric-learning losses"""

    def __init__(self, classes_num, embed_size, writer, soft_margin=False, loss_balancing=False):
        self.total_losses_num = 0
        self.writer = writer
        self.center_loss = CenterLoss(classes_num, embed_size, cos_dist=True)
        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)
        self.center_coeff = 0.0

        self.push_loss = PushLoss(0.7, soft_margin)
        self.push_loss_coeff = 0.0
        if self.push_loss_coeff > 0:
            self.total_losses_num += 1

        self.push_plus_loss = PushPlusLoss(0.7, soft_margin)
        self.push_plus_loss_coeff = 0.0
        if self.push_plus_loss_coeff > 0:
            self.total_losses_num += 1

        self.glob_push_plus_loss = GlobalPushPlus(0.7, soft_margin)
        self.glob_push_plus_loss_coeff = 0.0
        if self.glob_push_plus_loss_coeff > 0:
            self.total_losses_num += 1

        self.min_margin_loss = MinimumMargin(margin=.7)
        self.min_margin_loss_coeff = 0.0
        if self.min_margin_loss_coeff > 0:
            self.total_losses_num += 1

        self.loss_balancing = loss_balancing
        if self.loss_balancing:
            self.loss_weights = nn.Parameter(torch.FloatTensor(10).cuda())
            self.balancing_optimizer = torch.optim.SGD([self.loss_weights], lr=0.01)
            for i in range(self.total_losses_num):
                self.loss_weights.data[i] = 0.

    def _balance_losses(self, losses):
        assert len(losses) == self.total_losses_num
        for i, loss_val in enumerate(losses):
            losses[i] = torch.exp(-self.loss_weights[i]) * loss_val + \
                            0.5 * self.loss_weights[i]
        return sum(losses)

    def __call__(self, features, labels, epoch_num, iteration):
        log_string = ''
        all_loss_values = []
        center_loss_val = 0
        self.total_losses_num = 0
        if self.center_coeff > 0.:
            self.total_losses_num += 1
            center_loss_val = self.center_loss(features, labels)
            all_loss_values.append(center_loss_val)
            self.last_center_val = center_loss_val
            if self.writer is not None:
                self.writer.add_scalar('Loss/center_loss', center_loss_val, iteration)
            log_string += ' Center loss: %.4f' % center_loss_val

        push_loss_val = 0
        if self.push_loss_coeff > 0.0:
            self.total_losses_num += 1
            push_loss_val = self.push_loss(features, labels)
            all_loss_values.append(push_loss_val)
            if self.writer is not None:
                self.writer.add_scalar('Loss/push_loss', push_loss_val, iteration)
            log_string += ' Push loss: %.4f' % push_loss_val

        push_plus_loss_val = 0
        if self.push_plus_loss_coeff > 0.0 and self.center_coeff > 0.0:
            self.total_losses_num += 1
            push_plus_loss_val = self.push_plus_loss(features, self.center_loss.get_centers(), labels)
            all_loss_values.append(push_plus_loss_val)
            self.writer.add_scalar('Loss/push_plus_loss', push_plus_loss_val, iteration)
            log_string += ' Push Plus loss: %.4f' % push_plus_loss_val

        glob_push_plus_loss_val = 0
        if self.glob_push_plus_loss_coeff > 0.0 and self.center_coeff > 0.0:
            self.total_losses_num += 1
            glob_push_plus_loss_val = self.glob_push_plus_loss(features, self.center_loss.get_centers(), labels)
            all_loss_values.append(glob_push_plus_loss_val)
            if self.writer is not None:
                self.writer.add_scalar('Loss/global_push_plus_loss', glob_push_plus_loss_val, iteration)
            log_string += ' Global Push Plus loss: %.4f' % glob_push_plus_loss_val

        min_margin_loss_val = 0
        if self.min_margin_loss_coeff > 0.0 and self.center_coeff > 0.0:
            self.total_losses_num += 1
            min_margin_loss_val = self.min_margin_loss(self.center_loss.get_centers(), labels)
            all_loss_values.append(min_margin_loss_val)
            if self.writer is not None:
                self.writer.add_scalar('Loss/min_margin_loss', min_margin_loss_val, iteration)
            log_string += ' Min margin loss: %.4f' % min_margin_loss_val

        if self.loss_balancing and self.total_losses_num > 1:
            loss_value = self.center_coeff * self._balance_losses(all_loss_values)
            self.last_loss_value = loss_value
        else:
            loss_value = self.center_coeff * center_loss_val + self.push_loss_coeff * push_loss_val \
                        + self.push_plus_loss_coeff * push_plus_loss_val + self.min_margin_loss_coeff * min_margin_loss_val \
                        + self.glob_push_plus_loss_coeff * glob_push_plus_loss_val

        if self.total_losses_num > 0:
            if self.writer is not None:
                self.writer.add_scalar('Loss/AUX_losses', loss_value, iteration)

        return loss_value, log_string

    def init_iteration(self):
        """Initializes a training iteration"""
        if self.center_coeff > 0.:
            self.optimizer_centloss.zero_grad()

        if self.loss_balancing:
            self.balancing_optimizer.zero_grad()

    def end_iteration(self):
        """Finalizes a training iteration"""
        if self.loss_balancing and self.total_losses_num > 1:
            self.last_loss_value.backward(retain_graph=True)
            self.balancing_optimizer.step()

        if self.center_coeff > 0.:
            self.last_center_val.backward(retain_graph=True)
            for param in self.center_loss.parameters():
                param.grad.data *= (1. / self.center_coeff)
            self.optimizer_centloss.step()
