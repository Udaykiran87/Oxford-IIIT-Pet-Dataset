import logging
import sys
import os
import wget
import subprocess
from src.config import Configuration
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig
from src.utils import Model

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Custom_Train:
    def __init__(self, tfod_path_config=PathConfig):
        logger.info(">>>>>TFOD Custom Model Training preparation started<<<<<<")
        self.path_config = tfod_path_config
        self.model = Model()

    def train_model(self):
        self.model.download_model(self.path_config)


if __name__ == "__main__":
    project_config = Configuration()
    custom_train = Custom_Train(project_config.get_paths_config())
    custom_train.train_model()
