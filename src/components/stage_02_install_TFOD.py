import logging
import sys
import os
import wget
import subprocess
from subprocess import Popen, PIPE, STDOUT
from src.config import Configuration
from src.utils import create_directories
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


class TFOD_Install:
    def __init__(self, tfod_path_config=PathConfig):
        logger.info(">>>>>TFOD Installation started<<<<<<")
        self.path_config = tfod_path_config

    def install(self) -> None:
        try:
            if not os.path.exists(
                os.path.join(
                    self.path_config.apimodel_path, "research", "object_detection"
                )
            ):
                logger.info(
                    f">>>>>Git clone from 'https://github.com/tensorflow/models' started<<<<<<"
                )
                clone_cmd = f"git clone https://github.com/tensorflow/models {self.path_config.apimodel_path}"
                process = Popen(clone_cmd, stdout=PIPE, stderr=STDOUT)
                outdata, errdata = process.communicate()
                logger.info(
                    f"Output of git clone: {outdata}. Error data if any: {errdata}"
                )
            else:
                logger.info(
                    f">>>>>Git clone is not required as already cloned in the model directory<<<<<<"
                )

            # Install Tensorflow Object Detection
            logger.info(f">>>>>>TFOD installation Started<<<<<<<")
            if os.name == "posix":
                logger.info(f">>>>>>Posix System<<<<<<<")
                # os.popen("sh scripts/tfod_install_posix.sh")
                cmd = f"apt-get install protobuf-compiler"
                display1 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install ."
                display2 = subprocess.check_output(cmd, shell=True).decode()
                display = display1 + display2
                logger.info(f"{display}")
            if os.name == "nt":
                logger.info(f">>>>>>Nt System<<<<<<<")
                # os.popen("sh scripts/tfod_install_nt.sh")
                url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protoc-3.20.1-win64.zip"
                wget.download(url)
                cmd = f"move protoc-3.20.1-win64.zip {self.path_config.protoc_path}"
                display1 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"cd {self.path_config.protoc_path} && unzip protoc-3.20.1-win64.zip"
                display2 = subprocess.check_output(cmd, shell=True).decode()
                os.environ["PATH"] += os.pathsep + os.path.abspath(
                    os.path.join(f"{self.path_config.protoc_path}", "bin")
                )
                # cmd = f"cd {self.path_config.apimodel_path} && cd research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install"
                cmd = f"cd {self.path_config.apimodel_path} && cd research && protoc object_detection/protos/*.proto --python_out=."
                display3 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"cd {self.path_config.apimodel_path} && cd research/slim && pip install -e ."
                display4 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"pip uninstall protobuf -y"
                display5 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"pip install protobuf==3.15.7"
                display6 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"pip install pyparsing==2.4.7"
                display7 = subprocess.check_output(cmd, shell=True).decode()
                cmd = f"pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
                display8 = subprocess.check_output(cmd, shell=True).decode()
                # cmd = f"cd {self.path_config.apimodel_path} && cd research && copy object_detection/packages/tf2/setup.py setup.py && python setup.py build && python setup.py install"
                cmd = f"cd {self.path_config.apimodel_path} && cd research && cp object_detection/packages/tf2/setup.py . && python -m pip install ."
                display9 = subprocess.check_output(cmd, shell=True).decode()
                VERIFICATION_SCRIPT = os.path.join(
                    self.path_config.apimodel_path,
                    "research",
                    "object_detection",
                    "builders",
                    "model_builder_tf2_test.py",
                )
                # Verify Installation
                cmd = f"python {VERIFICATION_SCRIPT} "
                display10 = subprocess.check_output(cmd, shell=True).decode()
                display = (
                    display1
                    + display2
                    + display3
                    + display4
                    + display5
                    + display6
                    + display7
                    + display8
                    + display9
                    + display10
                )
                logger.info(f"{display}")
            logger.info(f">>>>>>TFOD installation Finished<<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    tfod_install = TFOD_Install(project_config.get_paths_config())
    tfod_install.install()
