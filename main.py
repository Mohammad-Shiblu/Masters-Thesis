import json
import torch
from utils.help import setup_logger
from train.generic_trainer import UNetTrainer
from train.munet_trainer import MUnetTrainer


def main():
    with open("config/train.json") as f:
        config = json.load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger = setup_logger(config)
    logger.info("Starting Training  with config:")
    for key, value in list(config.items())[1:]:
        logger.info(f"{key}: {value}")


    system = UNetTrainer(config= config, logger= logger, test_local=True)

    system.run()


if __name__ == "__main__":
    main()
