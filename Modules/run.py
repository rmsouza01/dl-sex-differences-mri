import hydra
from src import run_cross_validation_age


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    run_cross_validation_age(
        cfg.paths.images_path,
        cfg.paths.folds_path,
        cfg.paths.out_path,
        cfg.files.train_log,
        cfg.files.best_model_prefix,
        cfg.params.lr,
        cfg.params.epochs,
        cfg.params.batch_size,
    )

    return


if __name__ == "__main__":
    main()
