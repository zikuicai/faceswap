import keras.backend as K
from networks.faceswap_gan_model import FaceswapGANModel
from converter.video_converter import VideoConverter
from detector.face_detector import MTCNNFaceDetector

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,help='the path of the input video',required=True)
args = parser.parse_args()

input_fn = args.input


K.set_learning_phase(0)
# Input/Output resolution
RESOLUTION = 64 # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard" # standard, lite


model = FaceswapGANModel(**arch_config)
model.load_weights(path="./models")

fd = MTCNNFaceDetector(sess=K.get_session(), model_path="./mtcnn_weights/")
vc = VideoConverter()

vc.set_face_detector(fd)
vc.set_gan_model(model)

options = {
    # ===== Fixed =====
    "use_smoothed_bbox": True,
    "use_kalman_filter": True,
    "use_auto_downscaling": False,
    "bbox_moving_avg_coef": 0.65,
    "min_face_area": 35 * 35,
    "IMAGE_SHAPE": model.IMAGE_SHAPE,
    # ===== Tunable =====
    "kf_noise_coef": 3e-3,
    "use_color_correction": "hist_match",
    "detec_threshold": 0.7,
    "roi_coverage": 0.9,
    "enhance": 0.,
    "output_type": 3,
    "direction": "AtoB",
}


output_fn = input_fn.split('.')[0]+"_converted.mp4"
duration = None

vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, duration=duration)
