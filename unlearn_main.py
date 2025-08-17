import unlearn_strategies
from src import utils
from configs import unlearn_config

def main() -> None:
    args = unlearn_config.arguments
    # Set seed
    utils.set_seed(seed=args.seed)
    unlearn_strategies.unlearn_func.unlearn(args=args)


if __name__ == "__main__":
    main()