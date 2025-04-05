import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm # For progress bar

#------------------------------------
# Trainer Class
#------------------------------------

class Trainer:
    def __init__(self, model, config, train_dataset, val_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config['device']
        self.optimizer = None
        self.scaler = GradScaler(enabled=config['use_amp'])
        self.iter_num = 0
        self.best_val_loss = float('inf')

        # Ensure the save directory exists
        os.makedirs(config['drive_save_dir'], exist_ok=True)

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device == 'cuda' else False
        )

    @torch.no_grad()
    def estimate_loss(self):
        """ Estimates loss on train and validation sets. """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            if split == 'val' and self.val_dataset is None: continue
            dataset = self.train_dataset if split == 'train' else self.val_dataset
            loader = self._get_dataloader(dataset, shuffle=False) # No shuffle for eval
            losses = torch.zeros(self.config['eval_iters'])
            pbar = tqdm(enumerate(loader), total=self.config['eval_iters'],
                        desc=f"Evaluating {split}", leave=False)

            for k, (X, Y) in pbar:
                if k >= self.config['eval_iters']: break # Evaluate only eval_iters batches
                X, Y = X.to(self.device), Y.to(self.device)
                with autocast(enabled=self.config['use_amp']):
                    logits, loss = self.model(X, Y)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: NaN/Inf loss in {split} eval step {k}. Using high value.")
                    losses[k] = 50.0 # Assign high loss
                else:
                    losses[k] = loss.item()
                pbar.set_postfix({'loss': losses[k].item()})

            # Filter out any potential high values assigned due to NaN/Inf
            valid_losses = losses[losses < 49.0]
            if len(valid_losses) > 0:
                 out[split] = valid_losses.mean()
            elif len(losses) > 0: # If all were NaN/Inf, report the first high value
                 out[split] = losses[0]
            else: # Should not happen if eval_iters > 0
                 out[split] = float('inf')

        self.model.train()
        return out

    def save_checkpoint(self, is_best=False):
        """ Saves model checkpoint. """
        filename = "best_model.pth" if is_best else "latest_model.pth"
        filepath = os.path.join(self.config['drive_save_dir'], filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config, # Save config used for this checkpoint
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        print(f"Saving {'best' if is_best else 'latest'} checkpoint to {filepath}...")
        torch.save(checkpoint, filepath)
        print("Checkpoint saved.")

    def load_checkpoint(self, path=None):
        """ Loads model checkpoint. Tries 'best' then 'latest' if path is None. """
        if path is None:
            best_path = os.path.join(self.config['drive_save_dir'], "best_model.pth")
            latest_path = os.path.join(self.config['drive_save_dir'], "latest_model.pth")
            if os.path.exists(best_path):
                path = best_path
            elif os.path.exists(latest_path):
                path = latest_path
            else:
                print("No checkpoint found to load.")
                return False

        if not os.path.exists(path):
             print(f"Checkpoint file not found: {path}")
             return False

        print(f"Loading checkpoint from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Check for major config mismatches (optional but recommended)
            loaded_config = checkpoint.get('config', {})
            mismatches = []
            for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 'level']:
                if key in loaded_config and key in self.config and loaded_config[key] != self.config[key]:
                    mismatches.append(f"{key} (chkpt: {loaded_config[key]}, current: {self.config[key]})")
            if mismatches:
                print("Warning: Config mismatches detected between checkpoint and current settings:")
                for m in mismatches: print(f"  - {m}")
                # Decide whether to proceed or raise error based on mismatch severity

            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Reinitialize optimizer or load state? Loading is better for resumed runs.
            self.configure_optimizer() # Reinitialize for simplicity, or load below
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.iter_num = checkpoint.get('iter_num', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Checkpoint loaded. Resuming from iteration {self.iter_num}. Best val loss: {self.best_val_loss:.4f}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            # Ensure optimizer is initialized even if loading fails
            if self.optimizer is None: self.configure_optimizer()
            return False


    def configure_optimizer(self):
         """ Configures the AdamW optimizer. """
         # Standard AdamW optimizer
         self.optimizer = torch.optim.AdamW(
             self.model.parameters(),
             lr=self.config['learning_rate'],
             weight_decay=0.01 # Common weight decay value
         )


    def train(self):
        """ Runs the training loop. """
        train_loader = self._get_dataloader(self.train_dataset, shuffle=True)
        self.model.to(self.device)
        self.configure_optimizer() # Initialize optimizer

        # Attempt to load checkpoint
        self.load_checkpoint()

        # Use an iterator to manually handle stepping through data
        data_iter = iter(train_loader)
        start_time = time.time()

        print(f"Starting training from iteration {self.iter_num}...")
        self.model.train() # Ensure model is in training mode

        pbar = tqdm(range(self.iter_num, self.config['max_iters']), desc="Training Progress")
        for self.iter_num in pbar:

            # Evaluation step
            if self.iter_num > 0 and self.iter_num % self.config['eval_interval'] == 0:
                eval_start = time.time()
                losses = self.estimate_loss()
                eval_time = time.time() - eval_start
                print(f"\nStep {self.iter_num}: Train loss {losses.get('train', float('nan')):.4f}, "
                      f"Val loss {losses.get('val', float('nan')):.4f}, Eval time: {eval_time:.2f}s")

                current_val_loss = losses.get('val', float('inf')) # Use inf if no validation
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.save_checkpoint(is_best=True)
                else:
                    # Save latest checkpoint periodically even if not best
                    if self.iter_num % (self.config['eval_interval'] * 5) == 0: # e.g., every 5 evals
                         self.save_checkpoint(is_best=False)


            # Fetch next batch
            try:
                X, Y = next(data_iter)
            except StopIteration:
                # Epoch finished, re-create iterator
                data_iter = iter(train_loader)
                X, Y = next(data_iter)

            X, Y = X.to(self.device), Y.to(self.device)

            # Forward and backward pass with AMP
            with autocast(enabled=self.config['use_amp']):
                logits, loss = self.model(X, Y)

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"\nWarning: NaN/Inf loss detected at iter {self.iter_num}. Skipping step.")
                 self.optimizer.zero_grad(set_to_none=True) # Zero grads even if skipping
                 continue # Skip backward/step/update

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.config['grad_clip'] > 0:
                 # Unscale gradients before clipping
                 self.scaler.unscale_(self.optimizer)
                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update progress bar postfix
            pbar.set_postfix({'loss': loss.item():.4f, 'best_val': self.best_val_loss:.4f})

            # Termination condition
            if self.iter_num >= self.config['max_iters'] -1 :
                 print("\nMax iterations reached.")
                 break

        end_time = time.time()
        print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
        print(f"Final iteration: {self.iter_num}, Best validation loss: {self.best_val_loss:.4f}")
        # Save final latest model
        self.save_checkpoint(is_best=False)