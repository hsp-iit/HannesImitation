import torch
import numpy as np
from tqdm import tqdm


# diffusion policy imports
from hannes_imitation.external.diffusion_policy.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from hannes_imitation.external.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply


def validate(policy, vl_dataloader, device):
    vl_batch_losses = np.zeros(len(vl_dataloader))

    with torch.no_grad():
        vl_dataloader_iterator = tqdm(iterable=vl_dataloader, desc='Validation', leave=False)
        for i, batch in enumerate(vl_dataloader_iterator):
            batch = dict_apply(batch, lambda x: x.to(device))

            # compute average loss in minibatch
            loss = policy.compute_loss(batch)
            vl_batch_losses[i] = loss.item()
    
    mean_vl_loss = np.mean(vl_batch_losses)

    return mean_vl_loss


def evaluate_action_error(policy, dataloader, device):
    # sample trajectory from training set, and evaluate difference
    action_errors = np.zeros(len(dataloader))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device))

            # extract observation and ground truth action (gt)
            obs_dict = batch['obs']
            gt_action = batch['action'] # (B, horizon, Da)
            
            # predict actions (results are in original scale)
            result = policy.predict_action(obs_dict)
            
            pred_action = result['action_pred'] # (B, horizon, Da)

            # compute action error (mean absolute error)
            mae = torch.nn.functional.l1_loss(pred_action, gt_action)
            action_errors[i] = mae.item()
    
    mean_action_error = np.mean(action_errors)

    return mean_action_error


class TrainerDiffusionPolicy:

    def __init__(self, 
                 policy: DiffusionUnetImagePolicy,
                 optimizer: torch.optim.Optimizer,
                 normalizer,
                 tr_dataloader: torch.utils.data.DataLoader,
                 vl_dataloader: torch.utils.data.DataLoader = None,
                 learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 ):
        
        self.policy = policy
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.tr_dataloader = tr_dataloader
        self.vl_dataloader = vl_dataloader
        self.learning_rate_scheduler = learning_rate_scheduler

        # set policy normalizer
        policy.set_normalizer(normalizer=self.normalizer)


    def run(self, num_epochs, device):
        # device transfer
        _ = self.policy.to(device)
        _ = self.policy.normalizer.to(device)

        history = {'epoch': list(), 'tr_loss': list(), 'vl_loss': list(), 'vl_action_error': list()}

        # Training loop
        epoch_iterator = tqdm(iterable=range(num_epochs), desc="Epoch")
        for epoch in epoch_iterator:
            # train for one epoch
            tr_loss = self._train_epoch(device)
            
            # End of epoch, validate on validation set
            vl_loss = validate(self.policy, self.vl_dataloader, device)
            vl_action_error = evaluate_action_error(self.policy, self.vl_dataloader, device)

            # save training epoch results 
            history['epoch'].append(epoch + 1)
            history['tr_loss'].append(tr_loss)
            history['vl_loss'].append(vl_loss)
            history['vl_action_error'].append(vl_action_error)

            # log training epoch results
            postfix = dict(tr_loss=tr_loss, vl_loss=vl_loss, vl_action_error=vl_action_error)
            epoch_iterator.set_postfix(ordered_dict=postfix)

        return history

    def _train_epoch(self, device):
        # hold average loss for each mini-batch
        tr_loss_epoch = np.zeros(len(self.tr_dataloader))

        for i, batch in enumerate(self.tr_dataloader):
            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device))

            # compute average loss in minibatch
            # NOTE unused observations are discarded within
            loss = self.policy.compute_loss(batch)

            # compute loss gradient of minibatch
            loss.backward()

            # optimize
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.learning_rate_scheduler:
                self.learning_rate_scheduler.step() # step lr scheduler every batch. This is different from standard pytorch behavior
                
            # save loss batch
            tr_loss_epoch[i] = loss.item()
        
        # average loss over all minibatches
        mean_tr_loss_epoch = np.mean(tr_loss_epoch)

        return mean_tr_loss_epoch