import logging
import sys
import os
import wget
import subprocess
from src.config import Configuration
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig
from src.utils import Model

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Custom_Train:
    def __init__(self, tfod_path_config=PathConfig, tfod_url_name_config=UrlNameConfig):
        logger.info(">>>>>TFOD Custom Model Training preparation started<<<<<<")
        self.path_config = tfod_path_config
        self.url_name_config = tfod_url_name_config
        self.model = Model()

    def train_model(self):
        self.model.download_model(self.path_config, self.url_name_config)
        self.model.create_tf_record(self.path_config)
        self.model.copy_model_config_to_train_folder(
            self.path_config, self.url_name_config
        )


if __name__ == "__main__":
    project_config = Configuration()
    custom_train = Custom_Train(
        project_config.get_paths_config(), project_config.get_url_name_config()
    )
    custom_train.train_model()
