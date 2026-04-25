import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
from data.dataset import MyDataset
from models.convnext import ConvNext
from train import Trainer

class Config:
    # 基础配置
    train_json = '/mnt/data0/liuxin/Dataset/datasets_json/Fake_Hongkong/sample.json'
    val_json = '/mnt/data0/liuxin/Dataset/datasets_json/Fake_Hongkong/sample.json'
    save_dir = './checkpoints/test'
    
    # 训练超参数
    batch_size = 32  # 注意：这是每张卡的 batch_size
    lr = 1e-4
    epochs = 20
    image_size = 224

def main():
    # 初始化分布式环境 
    dist.init_process_group(backend="nccl")
    
    # 获取当前进程的 local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    cfg = Config()

    train_ds = MyDataset(cfg.train_json, image_size=cfg.image_size)
    val_ds = MyDataset(cfg.val_json, image_size=cfg.image_size)
    ckpt_path = "/mnt/data0/liuxin/ForensicHub-master/ForensicHub/log/ConvNextSmall_on_OSTF_withnot/checkpoint-19.pth"

    # 使用预训练的 ConvNext-Small
    model = ConvNext(pretrained=True, num_classes=1,checkpoint_path=ckpt_path)

    #实例化 Trainer 并启动训练
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg
    )

    trainer.fit()

    #清理进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    #test