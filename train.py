import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.fabric import Fabric


def main(cfg: DictConfig) -> None:
    fabric_args = cfg.fabric if cfg.fabric else {}
    fabric = Fabric(**fabric_args)
    fabric.launch()

    device = fabric.device
    print("device: ", device)


@hydra.main(version_base=None, config_path="config", config_name="config")
def training(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    main(cfg)

if __name__ == "__main__":
    training()
