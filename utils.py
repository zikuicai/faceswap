import numpy as np
import cv2
import os
import yaml


def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png")]


def load_images(image_paths, convert=None):
    iter_all_images = (cv2.resize(cv2.imread(fn), (256,256)) for fn in image_paths)
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = np.empty((len(image_paths),) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images


def get_transpose_axes( n ):
    if n % 2 == 0:
        y_axes = list(range(1, n-1, 2))
        x_axes = list(range(0, n-1, 2))
    else:
        y_axes = list(range(0, n-1, 2))
        x_axes = list(range(1, n-1, 2))
    return y_axes, x_axes, [n-1]


def stack_images(images):
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes = np.concatenate(new_axes)
        ).reshape(new_shape)


def get_G(test_A, test_B, path_A, path_B, batchSize):
    figure_A = np.stack([
        test_A,
        np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
        np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
        ], axis=1 )
    figure_B = np.stack([
        test_B,
        np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
        np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
        ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    return figure


def get_G_mask(test_A, test_B, path_A, path_B, batchSize):
    figure_A = np.stack([
        test_A,
        (np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
        (np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
        ], axis=1 )
    figure_B = np.stack([
        test_B,
        (np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
        (np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
        ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    return figure


def show_loss_config(loss_config):
    """
    Print out loss configuration. Called in loss function automation.
    
    Argument:
        loss_config: A dictionary. Configuration regarding the optimization.
    """
    for config, value in loss_config.items():
        print(f"{config} = {value}")


def make_html(filesets,img_dir,step_count):
    # this function makes a HTML file showing all the generated images in a webpage
    index_path = os.path.join(img_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>step</th><th>Transformed</th><th>Masks</th><th>Reconstruction</th></tr>")

    index.write("<tr>")
    index.write("<td>%s</td>" % step_count)
    for kind in range(3):
        index.write(f"<td><img src='{filesets[kind]}'></td>")
    index.write("</tr>")
    return index_path


def save_preview_image(test_A, test_B, 
                       path_A, path_B, 
                       path_bgr_A, path_bgr_B,
                       path_mask_A, path_mask_B, 
                       batchSize, save_fn="preview.jpg"):
    figure_A = np.stack([
        test_A,
        np.squeeze(np.array([path_bgr_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
        (np.squeeze(np.array([path_mask_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
        np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
        ], axis=1 )
    figure_B = np.stack([
        test_B,
        np.squeeze(np.array([path_bgr_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
        (np.squeeze(np.array([path_mask_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
        np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
        ], axis=1 )

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    cv2.imwrite(save_fn, figure)  


def load_yaml(path_configs):
    with open(path_configs, 'r') as f:
         return yaml.load(f)        




