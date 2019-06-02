import numpy as np
import cv2
from umeyama import umeyama
from scipy import ndimage
from pathlib import PurePath, Path

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'shift_range': 0.05,
    'random_flip': 0.5,
    }


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h,w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w//2,h//2), rotation, scale)
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine(image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:,::-1]
    return result


def random_warp_rev(image, res=64):
    assert image.shape == (256,256,6)
    res_scale = res//64
    assert res_scale >= 1, f"Resolution should be >= 64. Recieved {res}."
    interp_param = 80 * res_scale
    interp_slice = slice(interp_param//10,9*interp_param//10)
    dst_pnts_slice = slice(0,65*res_scale,16*res_scale)
    
    rand_coverage = np.random.randint(20) + 78 # random warping coverage
    rand_scale = np.random.uniform(5., 6.2) # random warping scale
    
    range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
    mapx = np.broadcast_to(range_, (5,5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5,5), scale=rand_scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=rand_scale)
    interp_mapx = cv2.resize(mapx, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
    interp_mapy = cv2.resize(mapy, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[dst_pnts_slice,dst_pnts_slice].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (res,res))
    return warped_image, target_image


def read_image(fn, fns_all_trn_data, res=64, random_transform_args=random_transform_args):

    # https://github.com/tensorflow/tensorflow/issues/5552
    # TensorFlow converts str to bytes in most places, including sess.run().
    if type(fn) == type(b"bytes"):
        fn = fn.decode("utf-8")
        fns_all_trn_data = [fn_all.decode("utf-8") for fn_all in fns_all_trn_data]
    
    raw_fn = PurePath(fn).parts[-1]
    image = cv2.imread(fn)
    if image is None:
        print(f"Failed reading image {fn}.")
        raise IOError(f"Failed reading image {fn}.")        
    
    image = cv2.resize(image, (256,256)) / 255 * 2 - 1

    bm_eyes = np.zeros_like(image)
    
    image = np.concatenate([image, bm_eyes], axis=-1)
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp_rev(image, res=res)
    
    bm_eyes = target_img[...,3:]
    warped_img = warped_img[...,:3]
    target_img = target_img[...,:3]
    
    warped_img, target_img, bm_eyes = \
    warped_img.astype(np.float32), target_img.astype(np.float32), bm_eyes.astype(np.float32)
    
    return warped_img, target_img, bm_eyes
