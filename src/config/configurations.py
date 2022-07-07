import os
import logging
import sys

from src.utils import read_yaml
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Configuration:
    def __init__(self):
        try:
            logger.info(">>>>>>Reading Config file<<<<<<")
            self.config = read_yaml(path_to_yaml=CONFIG_FILE_NAME)
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def get_paths_config(self) -> PathConfig:
        """This function reads all directory configurations.
        Returns: PathConfig object.
        """
        try:
            workspace_dir = self.config[PATH_KEY][WORKSPACE_NAME]
            scripts_dir = self.config[PATH_KEY][SCRIPTS_NAME]
            apimodel_dir = self.config[PATH_KEY][APIMODEL_NAME]
            annotation_dir = self.config[PATH_KEY][ANNOTATION_NAME]
            image_dir = self.config[PATH_KEY][IMAGE_NAME]
            model_dir = self.config[PATH_KEY][MODEL_NAME]
            pretrained_model_dir = self.config[PATH_KEY][PRETRAINED_MODEL_NAME]
            output_dir = self.config[PATH_KEY][OUTPUT_NAME]
            tfjs_dir = self.config[PATH_KEY][TFJS_NAME]
            tflite_dir = self.config[PATH_KEY][TFLITE_NAME]
            protoc_dir = self.config[PATH_KEY][PROTOC_NAME]

            artifcats_path = self.config[ARTIFACTS_KEY][ARTIFACTS_DIR]
            tensorflow_path = self.config[ARTIFACTS_KEY][TENSORFLOW_DIR]
            checkpoint_dir = self.config[ARTIFACTS_KEY][CUSTOM_MODEL_NAME]
            workspace_path = os.path.join(
                artifcats_path, tensorflow_path, workspace_dir
            )
            scripts_path = os.path.join(artifcats_path, tensorflow_path, scripts_dir)
            apimodel_path = os.path.join(artifcats_path, tensorflow_path, apimodel_dir)
            annotation_path = os.path.join(workspace_path, annotation_dir)
            image_path = os.path.join(workspace_path, image_dir)
            model_path = os.path.join(workspace_path, model_dir)
            pretrained_model_path = os.path.join(workspace_path, pretrained_model_dir)
            checkpoint_path = os.path.join(model_path, checkpoint_dir)
            output_path = os.path.join(checkpoint_path, output_dir)
            tfjs_path = os.path.join(checkpoint_path, tfjs_dir)
            tflite_path = os.path.join(checkpoint_path, tflite_dir)
            protoc_path = os.path.join(artifcats_path, tensorflow_path, protoc_dir)

            tfod_path_config = PathConfig(
                workspace_path=workspace_path,
                scripts_path=scripts_path,
                apimodel_path=apimodel_path,
                annotation_path=annotation_path,
                image_path=image_path,
                model_path=model_path,
                pretrained_model_path=pretrained_model_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                tfjs_path=tfjs_path,
                tflite_path=tflite_path,
                protoc_path=protoc_path,
            )
            return tfod_path_config
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = Configuration()
