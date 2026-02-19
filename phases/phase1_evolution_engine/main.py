from core.reproducibility.seed import set_global_seed
from core.logging.logger import setup_logger

from phases.phase1_evolution_engine.benchmarks.cifar10 import get_cifar10_loaders
from phases.phase1_evolution_engine.models.model_builder import BaselineCNN
from phases.phase1_evolution_engine.training.trainer import Trainer


def main():
    print("MAIN STARTED")

    set_global_seed(42)
    logger = setup_logger()
    logger.info("Starting Training")

    trainloader, testloader = get_cifar10_loaders(batch_size=128)

    model = BaselineCNN()
    trainer = Trainer(model)

    trainer.train(trainloader, epochs=3)
    trainer.evaluate(testloader)


if __name__ == "__main__":
    main()

