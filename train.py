import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        # 初始化最优损失记录
        self.best_loss = float('inf')

        # DDP
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # 训练集采样与加载
        self.train_sampler = DistributedSampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 验证集加载
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

    def train_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0
        #新加入的日志
        total_correct = 0
        total_samples = 0
        
        # 仅在主进程显示进度条
        if self.local_rank == 0:
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}")
        else:
            pbar = enumerate(self.train_loader)
            
        for i, batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            output = self.model(images, label=labels)
            loss = output['backward_loss']
            preds_prob = output['pred']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #新加入，计算当前batch的acc
            preds = (preds_prob > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item()
            if self.local_rank == 0:
                pbar.set_description(f"Epoch {epoch+1} Train Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        val_loss = torch.tensor(0.0).to(self.device)
        val_correct = torch.tensor(0).to(self.device)
        val_total = torch.tensor(0).to(self.device)

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            output = self.model(images, label=labels)
            val_loss += output['backward_loss'].item()
            preds_prob = output['pred']

            batch_loss = output['backward_loss'] # 从字典里拿到 Loss
            val_loss += batch_loss.item()        # 累加 Loss

            preds = (preds_prob > 0.5).long()
            val_correct += (preds == labels).sum()
            val_total += labels.size(0) 

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total, op=dist.ReduceOp.SUM)

        world_size = dist.get_world_size()
        avg_val_loss = val_loss.item() / (len(self.val_loader) * world_size)
        avg_acc = val_correct.item() / val_total.item()
        
        # 仅在主卡打印验证信息
        if self.local_rank == 0:
            print(f"✨ Epoch {epoch+1} 验证完成 | Loss: {avg_val_loss:.4f} | Accuracy: {100 * avg_acc:.2f}%")
        return avg_val_loss

    def fit(self):
        """执行完整的训练流程"""
        if self.local_rank == 0:
            print(f"Start Train, The main device: {self.device}")
            
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            dist.barrier() 

            if self.local_rank == 0:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    
                    if not os.path.exists(self.config.save_dir):
                        os.makedirs(self.config.save_dir)
                    
                    save_path = os.path.join(self.config.save_dir, "best_baseline.pth")
                    
                    torch.save(self.model.module.state_dict(), save_path)
                    print(f"--- Epoch {epoch+1}: 发现更优模型，权重已更新 ---")
                    #test