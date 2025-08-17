from src import utils
from fl_strategies import training
from configs import train_config


def main() -> None:
    # Set seed
    args = train_config.arguments
    utils.set_seed(seed=args.seed)
    fl = training.fl_training(arguments=args)
    fl.train()


if __name__ == "__main__":
    main()
