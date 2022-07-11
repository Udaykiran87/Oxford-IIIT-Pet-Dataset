import logging
import os
import sys
import wget
import subprocess
from src.exception import CustomException
from src.entity import PathConfig


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        pass

    @staticmethod
    def download_model(path_config, url_name_config):
        """
        This function downloads pre trained model.
        Returns: None
        """
        logger.info(">>>>>TFOD Model download started<<<<<<")
        try:
            if os.name == "posix":
                logger.info(f">>>>>>Posix System<<<<<<<")
                if not os.path.exists(
                    os.path.join(
                        path_config.pretrained_model_path,
                        url_name_config.pretrained_model_name + ".tar.gz",
                    )
                ):
                    logger.info(
                        f">>>>>Download of pre trained model {url_name_config.pretrained_model_name}.tar.gz started at {path_config.pretrained_model_path}<<<<<<"
                    )
                    wget.download(url_name_config.pretrained_model_url)
                    cmd = f"mv {url_name_config.pretrained_model_name+'.tar.gz'} {path_config.pretrained_model_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {path_config.pretrained_model_path} && tar -zxvf {url_name_config.pretrained_model_name+'.tar.gz'}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    logger.info(
                        f">>>>>Download of pre trained model {url_name_config.pretrained_model_name}.tar.gz finished at {path_config.pretrained_model_path}<<<<<<"
                    )
                else:
                    logger.info(
                        f">>>>>Download pre trained model not required as {url_name_config.pretrained_model_name}.tar.gz already present at {path_config.pretrained_model_path}<<<<<<"
                    )
            if os.name == "nt":
                logger.info(f">>>>>>Nt System<<<<<<<")
                if not os.path.exists(
                    os.path.join(
                        path_config.pretrained_model_path,
                        url_name_config.pretrained_model_name + ".tar.gz",
                    )
                ):
                    logger.info(
                        f">>>>>Download of pre trained model {url_name_config.pretrained_model_name}.tar.gz started at {path_config.pretrained_model_path}<<<<<<"
                    )
                    wget.download(url_name_config.pretrained_model_url)
                    cmd = f"move {url_name_config.pretrained_model_name+'.tar.gz'} {path_config.pretrained_model_path}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    cmd = f"cd {path_config.pretrained_model_path} && tar -zxvf {url_name_config.pretrained_model_name+'.tar.gz'}"
                    display = subprocess.check_output(cmd, shell=True).decode()
                    logger.info(f"{display}")
                    logger.info(
                        f">>>>>Download of pre trained model {url_name_config.pretrained_model_name}.tar.gz finished at {path_config.pretrained_model_path}<<<<<<"
                    )

                else:
                    logger.info(
                        f">>>>>Download pre trained model not required as {url_name_config.pretrained_model_name}.tar.gz already present at {path_config.pretrained_model_path}<<<<<<"
                    )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message

    @staticmethod
    def create_tf_record(path_config):
        """
        This function converts raw Oxford-IIIT Pet dataset into TFRecords.
        Returns: None
        """
        logger.info(">>>>>TFRecords Creation will start now<<<<<<")
        tfrecord_script = os.path.join(
            "object_detection",
            "dataset_tools",
            "create_pet_tf_record.py",
        )

        label_map_path = os.path.join("object_detection", "data", "pet_label_map.pbtxt")
        cmd = f"cd {path_config.apimodel_path} && cd research && python {tfrecord_script} --label_map_path={label_map_path} --data_dir={os.path.abspath(path_config.workspace_path)} --output_dir={os.path.abspath(path_config.tfrecords_path)}."
        display = subprocess.check_output(cmd, shell=True).decode()
        logger.info(f"{display}")
        logger.info(
            f"Train and validation tfrecords are created at {path_config.tfrecords_path}"
        )

    def copy_update_model_config(self):
        pass
