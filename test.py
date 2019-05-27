import keras.backend as K
from networks.faceswap_gan_model import FaceswapGANModel
from detector.face_detector import MTCNNFaceDetector
from converter.landmarks_alignment import *
from converter.face_transformer import FaceTransformer


# Input/Output resolution
RESOLUTION = 64  # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

# Path to training images
img_dirA = '../data/faces/faceA'
img_dirB = '../data/faces/faceB'
img_dirA_bm_eyes = "../data/binary_masks/faceA_eyes"
img_dirB_bm_eyes = "../data/binary_masks/faceB_eyes"

# Path to saved model weights
models_dir = "./models"

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard" # standard, lite


model = FaceswapGANModel(**arch_config)
model.load_weights(path=models_dir)


mtcnn_weights_dir = "./mtcnn_weights/"
fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)
ftrans = FaceTransformer()
ftrans.set_model(model)


# Read input image
input_img = cv2.imread("./jack.png")[..., :3]

if input_img.dtype == np.float32:
    print("input_img has dtype np.float32 (perhaps the image format is PNG). Scale it to uint8.")
    input_img = (input_img * 255).astype(np.uint8)

# Display input image
cv2.imshow('input_img', input_img)
cv2.waitKey(1)

# ----------------------------------------- detect face -----------------------------------------
face, lms = fd.detect_face(input_img)
if len(face) == 1:
    x0, y1, x1, y0, _ = face[0]
    det_face_im = input_img[int(x0):int(x1), int(y0):int(y1),:]
    try:
        src_landmarks = get_src_landmarks(x0, x1, y0, y1, lms)
        tar_landmarks = get_tar_landmarks(det_face_im)
        aligned_det_face_im = landmarks_match_mtcnn(det_face_im, src_landmarks, tar_landmarks)
    except:
        print("An error occured during face alignment.")
        aligned_det_face_im = det_face_im
elif len(face) == 0:
    raise ValueError("Error: no face detected.")
elif len(face) > 1:
    print (face)
    raise ValueError("Error: multiple faces detected")

cv2.imshow('aligned_det_face_im', aligned_det_face_im)
cv2.waitKey(1)


# Transform detected face
result_img, result_rgb, result_mask = ftrans.transform(
                    aligned_det_face_im, 
                    direction="AtoB", 
                    roi_coverage=0.93,
                    color_correction="adain_xyz",
                    IMAGE_SHAPE=(RESOLUTION, RESOLUTION, 3)
                    )
try:
    result_img = landmarks_match_mtcnn(result_img, tar_landmarks, src_landmarks)
    result_rgb = landmarks_match_mtcnn(result_rgb, tar_landmarks, src_landmarks)
    result_mask = landmarks_match_mtcnn(result_mask, tar_landmarks, src_landmarks)
except:
    print("An error occured during face alignment.")
    pass

result_input_img = input_img.copy()
result_input_img[int(x0):int(x1),int(y0):int(y1),:] = result_mask.astype(np.float32)/255*result_rgb +\
(1-result_mask.astype(np.float32)/255)*result_input_img[int(x0):int(x1),int(y0):int(y1),:]

# # Show result face
# cv2.imshow('result face', result_input_img[int(x0):int(x1), int(y0):int(y1), :])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Show transformed image before masking
# cv2.imshow('transformed image', result_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Show alpha mask
# cv2.imshow('alpha mask', result_mask[..., 0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Display interpolations before/after transformation
def interpolate_imgs(im1, im2):
    im1, im2 = map(np.float32, [im1, im2])
    out = [ratio * im1 + (1-ratio) * im2 for ratio in np.linspace(1, 0, 5)]
    out = map(np.uint8, out)
    return out


# plt.figure(figsize=(15,8))
cv2.imshow('final', np.hstack(interpolate_imgs(input_img, result_input_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()
