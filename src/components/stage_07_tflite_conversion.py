import logging
import os
import sys
import subprocess
from src.config import Configuration
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


class Tflite_Conversion:
    def __init__(self, tfod_path_config=PathConfig):
        logger.info(">>>>>Tflite model conversion<<<<<<")
        self.path_config = tfod_path_config

    def convert(self):
        try:
            logger.info(f">>>>>Tflite model conversion started<<<<<<")
            tflite_script = os.path.join(
                self.path_config.apimodel_path,
                "research",
                "object_detection",
                "export_tflite_graph_tf2.py ",
            )
            pipeline_config = os.path.join(
                self.path_config.checkpoint_path, "pipeline.config"
            )
            cmd = f"python {tflite_script} --pipeline_config_path={pipeline_config} --trained_checkpoint_dir={self.path_config.checkpoint_path} --output_directory={self.path_config.tflite_path}"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")

            frozen_tflite_path = os.path.join(
                self.path_config.tflite_path, "saved_model"
            )
            tflite_model = os.path.join(
                self.path_config.tflite_path, "saved_model", "detect.tflite"
            )
            cmd = f"tflite_convert \
            --saved_model_dir={frozen_tflite_path} \
            --output_file={tflite_model} \
            --input_shapes=1,300,300,3 \
            --input_arrays=normalized_input_image_tensor \
            --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
            --inference_type=FLOAT \
            --allow_custom_ops"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            logger.info(
                f">>>>>Tflite model is saved at {self.path_config.tflite_path}<<<<<<"
            )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    tflite = Tflite_Conversion(project_config.get_paths_config())
    tflite.convert()
