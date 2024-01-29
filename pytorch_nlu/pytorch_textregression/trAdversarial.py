# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/19 21:48
# @author  : Mo
# @function: adversarial of pytorch
# @reference: [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)


import torch.nn.functional as F
import torch


__all__ = ["FreeLB",
           "FGM",
           "PGD",
           "VAT"
           ]


class FreeLB:
    def __init__(self, adv_K, adv_lr, adv_init_mag, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag    # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:   # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化
        loss, logits = None, None
        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            inputs['input_ids'] = None
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)  # p='inf',无穷范数，获取绝对值最大者
                denorm = torch.clamp(denorm, min=1e-8)  # 类似np.clip，将数值夹逼到(min, max)之间
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # 计算该步的delta，然后累加到原delta值上(梯度上升)
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        return loss, logits


class FGM:
    def __init__(self, model, emb_name, epsilon=1.0):
        """
        Example
        # 初始化
        fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
        for batch_input, batch_label in data:
            # 正常训练
            loss = model(batch_input, batch_label)
            loss.backward() # 反向传播，得到正常的grad
            # 对抗训练
            fgm.attack() # 在embedding上添加对抗扰动
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
        """
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        """
        Example
        pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
        K = 3
        for batch_input, batch_label in data:
            # 正常训练
            loss = model(batch_input, batch_label)
            loss.backward() # 反向传播，得到正常的grad
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = model(batch_input, batch_label)
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
        """
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class VAT:
    """虚拟对抗训练 https://github.com/namisan/mt-dnn/blob/v0.2/alum/adv_masked_lm.py
    """

    def __init__(self, model, emb_name='word_embeddings', noise_var=1e-5, noise_gamma=1e-6, adv_step_size=1e-3,
                 adv_alpha=1, norm_type='l2', **kwargs):
        self.model = model
        self.noise_var = noise_var  # 噪声的方差
        self.noise_gamma = noise_gamma  # eps
        self.adv_step_size = adv_step_size  # 学习率
        self.adv_alpha = adv_alpha  # 对抗loss的权重
        self.norm_type = norm_type  # 归一化方式
        self.embed = None
        for (name, module) in self.model.named_modules():
            if emb_name in name:
                module.register_forward_hook(hook=self.hook)

    def hook(self, module, fea_in, fea_out):
        self.embed = fea_out
        return None

    def forward_(self, train_X, new_embed):
        # 把原来的train_X中的token_ids换成embedding形式
        if isinstance(train_X, (tuple, list)):
            new_train_X = [new_embed] + train_X[1:]
            adv_output = self.model.forward(
                *new_train_X) if self.model.forward.__code__.co_argcount >= 3 else self.model.forward(new_train_X)
        elif isinstance(train_X, torch.Tensor):
            adv_output = self.model.forward(new_embed)
        return adv_output

    def virtual_adversarial_training(self, train_X, logits):
        # 初始扰动 r
        noise = self.embed.data.new(self.embed.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()
        # x + r
        new_embed = self.embed.data.detach() + noise
        adv_output = self.forward_(train_X, new_embed)  # forward第一次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        # 梯度消失，退出
        if torch.isnan(norm) or torch.isinf(norm):
            return None
        # inner sum
        noise = noise + delta_grad * self.adv_step_size
        # projection
        noise = self.adv_project(noise, norm_type=self.norm_type, eps=self.noise_gamma)
        new_embed = self.embed.data.detach() + noise
        new_embed = new_embed.detach()
        # 在进行一次训练
        adv_output = self.forward_(train_X, new_embed)  # forward第二次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss_f = self.kl(adv_logits, logits.detach())
        adv_loss_b = self.kl(logits, adv_logits.detach())
        # 在预训练时设置为10，下游任务设置为1
        adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
        return adv_loss

    @staticmethod
    def kl(inputs, targets, reduction="sum"):
        """
        计算kl散度
        inputs：tensor，logits
        targets：tensor，logits
        """
        loss = F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction=reduction)
        return loss

    @staticmethod
    def adv_project(grad, norm_type='inf', eps=1e-6):
        """
        L0,L1,L2正则，对于扰动计算
        """
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction


