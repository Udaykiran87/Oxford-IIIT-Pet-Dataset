import logging
import os
import sys
import wget
import subprocess
from src.config import Configuration
from src.exception import CustomException
from src.entity import PathConfig


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Download_Dataset:
    def __init__(self, tfod_path_config=PathConfig):
        logger.info(">>>>>Download of OXFORD-III-PET-Dataset<<<<<<")
        self.path_config = tfod_path_config

    def download_data(self) -> None:
        """
        This function downloads OXFORD-III-PET-Dataset.
        Returns: None
        """
        logger.info(">>>>>Download of OXFORD-III-PET-Dataset started<<<<<<")
        try:
            if os.name == "posix":
                logger.info(f">>>>>>Posix System<<<<<<<")
                if not os.path.exists(
                    os.path.join(
                        self.path_config.workspace_path,
                        "images.tar.gz",
                    )
                ):
                    logger.info(
                        ">>>>>Download of OXFORD-III-PET-Dataset (Images) started<<<<<<"
                    )
                    wget.download(self.path_config.dataset_image_url)
                    cmd = f"mv images.tar.gz {self.path_config.workspace_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.workspace_path} && tar -xvf images.tar.gz"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                else:
                    logger.info(
                        ">>>>>Download not required as images.tar.gz already present at {self.path_config.workspace_path}<<<<<<"
                    )

                if not os.path.exists(
                    os.path.join(
                        self.path_config.workspace_path,
                        "annotations.tar.gz",
                    )
                ):
                    logger.info(
                        ">>>>>Download of OXFORD-III-PET-Dataset (Annotations) started<<<<<<"
                    )
                    wget.download(self.path_config.dataset_annotation_url)
                    cmd = f"mv annotations.tar.gz {self.path_config.workspace_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.workspace_path} && tar -xvf annotations.tar.gz"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                else:
                    logger.info(
                        ">>>>>Download not required as annotations.tar.gz already present at {self.path_config.workspace_path}<<<<<<"
                    )
            if os.name == "nt":
                logger.info(f">>>>>>Nt System<<<<<<<")
                if not os.path.exists(
                    os.path.join(
                        self.path_config.workspace_path,
                        "images.tar.gz",
                    )
                ):
                    logger.info(
                        ">>>>>Download of OXFORD-III-PET-Dataset (Images) started<<<<<<"
                    )
                    wget.download(self.path_config.dataset_image_url)
                    cmd = f"move images.tar.gz {self.path_config.workspace_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.workspace_path} && tar -xvf images.tar.gz"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    logger.info(
                        ">>>>>Download of OXFORD-III-PET-Dataset (Images) finished<<<<<<"
                    )
                else:
                    logger.info(
                        ">>>>>Download not required as images.tar.gz already present at {self.path_config.workspace_path}<<<<<<"
                    )

                if not os.path.exists(
                    os.path.join(
                        self.path_config.workspace_path,
                        "annotations.tar.gz",
                    )
                ):
                    logger.info(
                        ">>>>>Download of OXFORD-III-PET-Dataset (Annotations) started<<<<<<"
                    )
                    wget.download(self.path_config.dataset_annotation_url)
                    cmd = f"move annotations.tar.gz {self.path_config.workspace_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {self.path_config.workspace_path} && tar -xvf annotations.tar.gz"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    logger.info(
                        ">>>>>Download of OXFORD-III-PET-Dataset (Annotations) finished<<<<<<"
                    )
                else:
                    logger.info(
                        ">>>>>Download not required as annotations.tar.gz already present at {self.path_config.workspace_path}<<<<<<"
                    )

        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
        logger.info(">>>>>Download of OXFORD-III-PET-Dataset finished<<<<<<")


if __name__ == "__main__":
    project_config = Configuration()
    download = Download_Dataset(project_config.get_paths_config())
    download.download_data()
