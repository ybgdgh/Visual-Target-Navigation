#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.optim as optim
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO

class DDPPOSLAM(DDPPO):
    
    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                # print("***********************************************obs_batch: ", obs_batch.shape)
                # print("***********************************************obs_batch: ", len(obs_batch))
                # print("***********************************************actions_batch: ", actions_batch.shape)
                # print("***********************************************actions_batch: ", actions_batch)

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                # print("old_action_log_probs_batch: ", old_action_log_probs_batch)
                # print("action_log_probs: ", action_log_probs)

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                
                # print("adv_targ: ", adv_targ)
                # print("surr1: ", surr1)
                # print("surr2: ", surr2)
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                    # print("value_losses_clipped: ", value_losses_clipped)
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                # print("values: ", values)
                # print("value_preds_batch: ", value_preds_batch)
                # if return_batch.device == "cuda:0":
                # print("return_batch: ", return_batch)

                # print("value_loss: ", value_loss)
                # print("action_loss: ", action_loss)
                # print("total_loss: ", total_loss)

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
