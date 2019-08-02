import os
import random

import torch
import torch.backends.cudnn as cudnn

from config import get_config
config = get_config()
from dataloader import get_loader

if config.model_name == 'PrismaNet':
    from train_prisma import Trainer
elif config.model_name == 'PrismaMattingNet':
    from train_prisma_matting import Trainer
from test import Tester

def main(config):
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    data_loader = get_loader(config.data_path, config.batch_size, config.image_size,
                            shuffle=True, num_workers=int(config.workers))

    tester = Tester(config, data_loader)
    tester.test()

    # trainer = Trainer(config, data_loader)
    # trainer.train()
    

if __name__ == "__main__":
    config = get_config()
    main(config)
