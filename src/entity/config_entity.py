from collections import namedtuple

PathConfig = namedtuple(
    "PathConfig",
    [
        "workspace_path",
        "scripts_path",
        "apimodel_path",
        "annotation_path",
        "image_path",
        "model_path",
        "pretrained_model_path",
        "checkpoint_path",
        "output_path",
        "tfjs_path",
        "tflite_path",
        "protoc_path",
    ],
)