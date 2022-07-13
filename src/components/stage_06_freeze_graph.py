import logging
import os
import sys
import subprocess
from src.config import Configuration
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Freeze:
    def __init__(self, tfod_path_config=PathConfig):
        logger.info(">>>>>TFOD model graph freeze<<<<<<")
        self.path_config = tfod_path_config

    def freeze_graph(self):
        try:
            logger.info(f">>>>>Frozen model graph process started<<<<<<")
            freeze_script = os.path.join(
                self.path_config.apimodel_path,
                "research",
                "object_detection",
                "exporter_main_v2.py ",
            )
            pipeline_config = os.path.join(
                self.path_config.checkpoint_path, "pipeline.config"
            )
            cmd = f"python {freeze_script} --input_type=image_tensor --pipeline_config_path={pipeline_config} --trained_checkpoint_dir={self.path_config.checkpoint_path} --output_directory={self.path_config.output_path}"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            logger.info(
                f">>>>>Frozen model graph is saved at {self.path_config.output_path}<<<<<<"
            )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    freeze = Freeze(project_config.get_paths_config())
    freeze.freeze_graph()
