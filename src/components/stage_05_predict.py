import logging
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from src.config import Configuration
from src.constants import *
from src.exception import CustomException
from src.entity import PathConfig, UrlNameConfig
from src.utils import Model
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


@tf.function
def detect_fn(image, detection_model):
    """
    This function performs image detection on single image.
    Returns: None
    """
    try:
        logger.info(f">>>>>Object detection on single image started<<<<<<")
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    except Exception as e:
        message = CustomException(e, sys)
        logger.error(message.error_message)
        raise message


class Predict:
    def __init__(self, tfod_path_config=PathConfig, tfod_url_name_config=UrlNameConfig):
        logger.info(">>>>>Object Detection on Image<<<<<<")
        self.path_config = tfod_path_config
        self.url_name_config = tfod_url_name_config
        self.model = Model(tfod_path_config, tfod_url_name_config)
        self.loaded_model = self.model.load_model(ckpt_name="ckpt-3")
        self.category_index = label_map_util.create_category_index_from_labelmap(
            os.path.join(
                self.path_config.apimodel_path,
                "research",
                "object_detection",
                "data",
                "pet_label_map.pbtxt",
            )
        )

    def detection_from_image(self, image_path):
        """
        This function performs detection and visualization on single image.
        Returns: None
        """
        try:
            logger.info(
                f">>>>>Object Detection on Image {image_path} will start now<<<<<<"
            )

            img = cv2.imread(image_path)
            image_np = np.array(img)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32
            )
            detections = detect_fn(input_tensor, self.loaded_model)

            num_detections = int(detections.pop("num_detections"))
            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }
            detections["num_detections"] = num_detections

            # detection_classes should be ints.
            detections["detection_classes"] = detections["detection_classes"].astype(
                np.int64
            )

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections["detection_boxes"],
                detections["detection_classes"] + label_id_offset,
                detections["detection_scores"],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=0.35,
                agnostic_mode=False,
            )

            base_image = os.path.splitext(image_path)[0]
            save_image = os.path.join(base_image + "_out" + ".jpg")
            cv2.imwrite(save_image, image_np_with_detections)
            logger.info(
                f">>>>>Object Detection on Image is saved as {save_image}<<<<<<"
            )
            logger.info(f">>>>>Object Detection on Image {image_path} finished<<<<<<")
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message

    def detection_from_video(self, video_path):
        """
        This function performs detection and visualization on video.
        Returns: None
        """
        try:
            logger.info(
                f">>>>>Object Detection on video at {video_path} will start now<<<<<<"
            )
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while cap.isOpened():
                ret, frame = cap.read()
                image_np = np.array(frame)

                input_tensor = tf.convert_to_tensor(
                    np.expand_dims(image_np, 0), dtype=tf.float32
                )
                detections = detect_fn(input_tensor, self.loaded_model)

                num_detections = int(detections.pop("num_detections"))
                detections = {
                    key: value[0, :num_detections].numpy()
                    for key, value in detections.items()
                }
                detections["num_detections"] = num_detections

                # detection_classes should be ints.
                detections["detection_classes"] = detections[
                    "detection_classes"
                ].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections["detection_boxes"],
                    detections["detection_classes"] + label_id_offset,
                    detections["detection_scores"],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=3,
                    min_score_thresh=0.35,
                    agnostic_mode=False,
                )

                cv2.imshow(
                    "object detection", cv2.resize(image_np_with_detections, (800, 600))
                )

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            logger.info(
                f">>>>>Object Detection on video at {video_path} is finished<<<<<<"
            )
        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
            raise message


if __name__ == "__main__":
    project_config = Configuration()
    predict = Predict(
        project_config.get_paths_config(), project_config.get_url_name_config()
    )
    path_config = project_config.get_paths_config()
    image_path = os.path.join(
        path_config.test_images_path,
        "Abyssinian_10.jpg",
    )
    predict.detection_from_image(image_path)
    video_path = os.path.join(
        path_config.test_images_path,
        "cat_video.mp4",
    )
    predict.detection_from_video(video_path)
