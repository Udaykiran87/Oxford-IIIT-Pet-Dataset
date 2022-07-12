import logging
import os
import sys
import wget
import subprocess
import re
from src.exception import CustomException
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from src.entity import PathConfig, UrlNameConfig


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self, tfod_path_config=PathConfig, tfod_url_name_config=UrlNameConfig):
        self.path_config = tfod_path_config
        self.url_name_config = tfod_url_name_config
        self.config = config_util.get_configs_from_pipeline_file(
            os.path.join(self.path_config.checkpoint_path, "pipeline.config")
        )

    @staticmethod
    def download_model(path_config, url_name_config):
        """
        This function downloads pre trained model.
        Returns: None
        """
        try:
            logger.info(">>>>>TFOD Model download started<<<<<<")
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
        try:
            logger.info(">>>>>TFRecords Creation will start now<<<<<<")
            train_tfrecord_re = "pet_faces_train.record-"
            val_tfrecord_re = "pet_faces_val.record-"
            for filename in os.listdir(path_config.tfrecords_path):
                if re.search(train_tfrecord_re, filename):
                    train = True
                    break
            for filename in os.listdir(path_config.tfrecords_path):
                if re.search(val_tfrecord_re, filename):
                    val = True
                    break
            if train == True and val == True:
                logger.info(">>>>>TFRecords already present<<<<<<")
                return
            tfrecord_script = os.path.join(
                "object_detection",
                "dataset_tools",
                "create_pet_tf_record.py",
            )

            label_map_path = os.path.join(
                "object_detection", "data", "pet_label_map.pbtxt"
            )
            cmd = f"cd {path_config.apimodel_path} && cd research && python {tfrecord_script} --label_map_path={label_map_path} --data_dir={os.path.abspath(path_config.workspace_path)} --output_dir={os.path.abspath(path_config.tfrecords_path)}."
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            logger.info(
                f"Train and validation tfrecords are created at {path_config.tfrecords_path}"
            )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message

    @staticmethod
    def copy_model_config_to_train_folder(path_config, url_name_config):
        """
        This function copies model config file to train folder.
        Returns: None
        """
        try:
            logger.info(">>>>>Model config copy to train folder will start now<<<<<<")
            if os.path.exists(
                os.path.join(path_config.checkpoint_path, "pipeline.config")
            ):
                logger.info(
                    f"pipeline.config alreday present at {path_config.checkpoint_path}"
                )
                return
            if os.name == "posix":
                logger.info(f">>>>>>Posix System<<<<<<<")
                cmd = f"cp {os.path.join(path_config.pretrained_model_path, url_name_config.pretrained_model_name, 'pipeline.config')} {path_config.checkpoint_path}"
            if os.name == "nt":
                logger.info(f">>>>>>Nt System<<<<<<<")
                cmd = f"copy {os.path.join(path_config.pretrained_model_path, url_name_config.pretrained_model_name, 'pipeline.config')} {path_config.checkpoint_path}"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            logger.info(
                f"File: {os.path.join(path_config.pretrained_model_path, url_name_config.pretrained_model_name, 'pipeline.config')} copied to: {path_config.checkpoint_path}"
            )
            logger.info(f">>>>>>Model config copy to train folder finsihed<<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message

    def update_model_config(self):
        """
        This function updates the model configuration file for training.
        Returns: None
        """
        try:
            logger.info(f">>>>>Model config file update will start now<<<<<<")
            label_map_path = os.path.join(
                self.path_config.apimodel_path,
                "research",
                "object_detection",
                "data",
                "pet_label_map.pbtxt",
            )
            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(
                os.path.join(self.path_config.checkpoint_path, "pipeline.config"), "r"
            ) as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline_config)

            pipeline_config.model.ssd.num_classes = 37
            pipeline_config.train_config.batch_size = 16
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
                self.path_config.pretrained_model_path,
                self.url_name_config.pretrained_model_name,
                "checkpoint",
                "ckpt-0",
            )
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path = label_map_path
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
                os.path.join(
                    self.path_config.tfrecords_path,
                    "pet_faces_train.record-?????-of-00010",
                )
            ]
            pipeline_config.eval_input_reader[0].label_map_path = label_map_path
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[
                :
            ] = [
                os.path.join(
                    self.path_config.tfrecords_path,
                    "pet_faces_val.record-?????-of-00010",
                )
            ]
            config_text = text_format.MessageToString(pipeline_config)
            with tf.io.gfile.GFile(
                os.path.join(self.path_config.checkpoint_path, "pipeline.config"), "wb"
            ) as f:
                f.write(config_text)
            logger.info(
                f">>>>>Model config file update of {os.path.join(self.path_config.checkpoint_path, 'pipeline.config')} finished.<<<<<<"
            )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message

    def custom_model_train(self):
        """
        This function is used to perform custom model training.
        Returns: None
        """
        try:
            logger.info(f">>>>>Custom Model training started<<<<<<")
            TRAINING_SCRIPT = os.path.join(
                self.path_config.apimodel_path,
                "research",
                "object_detection",
                "model_main_tf2.py",
            )
            cmd = f"python {TRAINING_SCRIPT} --model_dir={self.path_config.checkpoint_path} --pipeline_config_path={os.path.join(self.path_config.checkpoint_path, 'pipeline.config')} --num_train_steps=2000"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            logger.info(
                f">>>>>Custom Model training finished and checkpoints are saved at {self.path_config.checkpoint_path}<<<<<<"
            )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message

    def model_eval(self):
        """
        This function is used to evaluate the trained model.
        Returns: None
        """
        try:
            logger.info(f">>>>>Custom Model evaluation started<<<<<<")
            EVALUATE_SCRIPT = os.path.join(
                self.path_config.apimodel_path,
                "research",
                "object_detection",
                "model_main_tf2.py",
            )
            cmd = f"python {EVALUATE_SCRIPT} --model_dir={self.path_config.checkpoint_path} --pipeline_config_path={os.path.join(self.path_config.checkpoint_path, 'pipeline.config')} --checkpoint_dir={self.path_config.checkpoint_path}"
            display = subprocess.check_output(cmd, shell=True).decode()
            logger.info(f"{display}")
            logger.info(f">>>>>Model evaluation finished<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message
