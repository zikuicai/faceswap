import os
import glob
import time
from pathlib import Path

import cv2
import tensorflow as tf
from keras.layers import *
import keras.backend as K

from data_loader.data_loader import DataLoader
from utils import get_G, get_G_mask, show_loss_config, make_html
from networks.faceswap_gan_model import FaceswapGANModel
# https://github.com/rcmalli/keras-vggface
#!pip install keras_vggface --no-dependencies
from keras_vggface.vggface import VGGFace


# ----------------------------------------- config -----------------------------------------

# Number of CPU cores, for parallelism
num_cpus = os.cpu_count()
# Input/Output resolution
RESOLUTION = 64 # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."
batchSize = 8
assert (batchSize != 1 and batchSize % 2 == 0) , "batchSize should be an even number."

# Use motion blurs (data augmentation)
# set True if training data contains images extracted from videos
use_da_motion_blur = True

# Use eye-aware training
# require images generated from prep_binary_masks.ipynb
use_bm_eyes = True

# Probability of random color matching (data augmentation)
prob_random_color_match = 0.5

da_config = {
    "prob_random_color_match": prob_random_color_match,
    "use_da_motion_blur": use_da_motion_blur,
    "use_bm_eyes": use_bm_eyes
}

# Path to training images
img_dirA = '../data/faces/faceA'
img_dirB = '../data/faces/faceB'
img_dirA_bm_eyes = "../data/binary_masks/faceA_eyes"
img_dirB_bm_eyes = "../data/binary_masks/faceB_eyes"

# Path to saved model weights
models_dir = "./models"
results_dir = "./results"
trans_dir = "./results/trans"
masks_dir = "./results/masks"
recon_dir = "./results/recon"
# Create ./models directory
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(trans_dir).mkdir(parents=True, exist_ok=True)
Path(masks_dir).mkdir(parents=True, exist_ok=True)
Path(recon_dir).mkdir(parents=True, exist_ok=True)


# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard" # standard, lite

# Loss function weights configuration
loss_weights = {}
loss_weights['w_D'] = 0.1 # Discriminator
loss_weights['w_recon'] = 1. # L1 reconstruction loss
loss_weights['w_edge'] = 0.1 # edge loss
loss_weights['w_eyes'] = 30. # reconstruction and edge loss on eyes area
loss_weights['w_pl'] = (0.01, 0.1, 0.3, 0.1) # perceptual loss (0.003, 0.03, 0.3, 0.3)

# Init. loss config.
loss_config = {}
loss_config["gan_training"] = "mixup_LSGAN" # "mixup_LSGAN" or "relativistic_avg_LSGAN"
loss_config['use_PL'] = False
loss_config["PL_before_activ"] = False
loss_config['use_mask_hinge_loss'] = False
loss_config['m_mask'] = 0.
loss_config['lr_factor'] = 1.
loss_config['use_cyclic_loss'] = False


# ----------------------------------------- load data -----------------------------------------

# Get filenames
train_A = glob.glob(img_dirA+"/*.*")
train_B = glob.glob(img_dirB+"/*.*")
train_AnB = train_A + train_B

assert len(train_A), "No image found in " + str(img_dirA)
assert len(train_B), "No image found in " + str(img_dirB)
print ("Number of images in folder A: " + str(len(train_A)))
print ("Number of images in folder B: " + str(len(train_B)))

if use_bm_eyes:
    assert len(glob.glob(img_dirA_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirA_bm_eyes)
    assert len(glob.glob(img_dirB_bm_eyes+"/*.*")), "No binary mask found in " + str(img_dirB_bm_eyes)
    assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A), \
    "Number of faceA images does not match number of their binary masks. Can be caused by any none image file in the folder."
    assert len(glob.glob(img_dirB_bm_eyes+"/*.*")) == len(train_B), \
    "Number of faceB images does not match number of their binary masks. Can be caused by any none image file in the folder."

train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes, 
                          RESOLUTION, num_cpus, K.get_session(), **da_config)
train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes, 
                          RESOLUTION, num_cpus, K.get_session(), **da_config)

# ----------------------------------------- define models -----------------------------------------

model = FaceswapGANModel(**arch_config)
model.load_weights(path=models_dir)

# VGGFace ResNet50
vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
vggface.summary()

model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
model.build_train_functions(loss_weights=loss_weights, **loss_config)


# ----------------------------------------- training -----------------------------------------

t0 = time.time()

# This try/except is meant to resume training that was accidentally interrupted
try:
    gen_iterations
    print(f"Resume training from iter {gen_iterations}.")
except:
    gen_iterations = 0

errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
errGAs = {}
errGBs = {}
# Dictionaries are ordered in Python 3.6
for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
    errGAs[k] = 0
    errGBs[k] = 0

display_iters = 300
backup_iters = 5000
TOTAL_ITERS = 40000
step_count = 0


def reset_session(save_path):
    global model, vggface, train_batchA, train_batchB
    model.save_weights(path=save_path)
    del model, vggface, train_batchA, train_batchB
    K.clear_session()
    model = FaceswapGANModel(**arch_config)
    model.load_weights(path=save_path)
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes, 
                              RESOLUTION, num_cpus, K.get_session(), **da_config)


def reconfig(stage):
    m_mask = [0.0, 0.5, 0.2, 0.4, 0., 0.1, 0.0]
    use_mask_hinge_loss = [False, True, True, True, False, True, False]
    lr_factor = [1., 1., 1., 1., 0.3, 0.3, 0.1]

    loss_config['use_PL'] = True
    loss_config['use_mask_hinge_loss'] = use_mask_hinge_loss[stage]
    loss_config['m_mask'] = m_mask[stage]
    loss_config['lr_factor'] = lr_factor[stage]

    reset_session(models_dir)
    print("Building new loss funcitons...")
    show_loss_config(loss_config)
    model.build_train_functions(loss_weights=loss_weights, **loss_config)
    print("Done.")


while gen_iterations <= TOTAL_ITERS: 
    
    if gen_iterations == 5:
        print ("working.")

    # Loss function automation
    if gen_iterations == int(0.2*TOTAL_ITERS): reconfig(0)
    elif gen_iterations == int(0.3*TOTAL_ITERS): reconfig(1)
    elif gen_iterations == int(0.4*TOTAL_ITERS): reconfig(2)
    elif gen_iterations == int(0.5*TOTAL_ITERS): reconfig(3)
    elif gen_iterations == int(0.7*TOTAL_ITERS): reconfig(4)
    elif gen_iterations == int(0.8*TOTAL_ITERS):
        model.decoder_A.load_weights("models/decoder_B.h5") # swap decoders
        model.decoder_B.load_weights("models/decoder_A.h5") # swap decoders
        reconfig(5)
    elif gen_iterations == int(0.9*TOTAL_ITERS): reconfig(6)
    
    gen_iterations+=1
    # Train dicriminators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
    errDA_sum +=errDA[0]
    errDB_sum +=errDB[0]
    # Train generators for one batch
    data_A = train_batchA.get_next_batch()
    data_B = train_batchB.get_next_batch()
    errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
    errGA_sum += errGA[0]
    errGB_sum += errGB[0]
    for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
        errGAs[k] += errGA[i]
        errGBs[k] += errGB[i]
    
    # Visualization
    if gen_iterations % display_iters == 0:
        # Display loss information
        show_loss_config(loss_config)
        print("----------") 
        print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
        % (gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, time.time()-t0))  
        print("----------") 
        print("Generator loss details:")
        print(f'[Adversarial loss]')  
        print(f'GA: {errGAs["adv"]/display_iters:.4f} GB: {errGBs["adv"]/display_iters:.4f}')
        print(f'[Reconstruction loss]')
        print(f'GA: {errGAs["recon"]/display_iters:.4f} GB: {errGBs["recon"]/display_iters:.4f}')
        print(f'[Edge loss]')
        print(f'GA: {errGAs["edge"]/display_iters:.4f} GB: {errGBs["edge"]/display_iters:.4f}')
        if loss_config['use_PL'] == True:
            print(f'[Perceptual loss]')
            try:
                print(f'GA: {errGAs["pl"][0]/display_iters:.4f} GB: {errGBs["pl"][0]/display_iters:.4f}')
            except:
                print(f'GA: {errGAs["pl"]/display_iters:.4f} GB: {errGBs["pl"]/display_iters:.4f}')
        
        # Display images
        print("----------")
        wA, tA, _ = train_batchA.get_next_batch()
        wB, tB, _ = train_batchB.get_next_batch()
        img_transform = get_G(tA, tB, model.path_A, model.path_B, batchSize)
        img_masks = get_G_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)
        img_reconstruct = get_G(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize)
        cv2.imwrite(os.path.join(trans_dir,str(step_count))+'.jpg',img_transform)
        cv2.imwrite(os.path.join(masks_dir,str(step_count))+'.jpg',img_masks)
        cv2.imwrite(os.path.join(recon_dir,str(step_count))+'.jpg',img_reconstruct)
        img_set = [f"trans/{step_count}.jpg",f"masks/{step_count}.jpg",f"recon/{step_count}.jpg"]
        make_html(img_set,img_dir=results_dir,step_count=step_count)

        errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
            errGAs[k] = 0
            errGBs[k] = 0

        step_count += 1
        # Save models
        model.save_weights(path=models_dir)
    
    # Backup models
    if gen_iterations % backup_iters == 0: 
        bkup_dir = f"{models_dir}/backup_iter{gen_iterations}"
        Path(bkup_dir).mkdir(parents=True, exist_ok=True)
        model.save_weights(path=bkup_dir)
