import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)


if __name__ == "__main__":
    main()
