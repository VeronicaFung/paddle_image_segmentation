import os
import paddle
from paddle.fluid.dygraph.base import to_variable
#from basic_model import BasicModel
from basic_dataloader import BasicDataLoader
import numpy as np
from PIL import Image
import paddle.fluid as fluid
import cv2
import tqdm
import argparse
from basic_data_preprocessing import TrainAugmentation
from deeplab import DeepLab
from pspnet import PSPNet
from unet import UNet



def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def save_blend_image(image_file, pred_file):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    return image
    #o_file = pred_file[0:-4] + "_blend.png"
    #image.save(o_file)




def inference_resize():
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=512, interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("interp should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise TypeError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, im_info=None, label=None):
        if im_info is None:
            im_info = list()
        im_info.append(('resize', im.shape[:2]))
        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = resize(label, self.target_size, cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)

#def inference_sliding():


def inference_multi_scale():
    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, im, im_info=None, label=None):
        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * im.shape[1]))
        h = int(round(scale_factor * im.shape[0]))

        im = resize(im, (w, h), cv2.INTER_LINEAR)
        if label is not None:
            label = resize(label, (w, h), cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


#def save_images(image_path, image):

#    cv2.imwrite(image_path, image)



def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def infer(model, test_dataset=None, model_dir=None, save_dir='output'):

    model.eval()
    pred_saved_dir = os.path.join(save_dir, 'prediction/')

    print("Start to predict...")
    for i,data in enumerate(test_dataset):
        image=data[0]
        label=data[1]
        prev_shape =(label.shape[1],label.shape[2])
        image=image.astype(np.float32)
        label=label.astype(np.int64)
        image=fluid.layers.transpose(image, perm=[0,3,1,2])
        image = to_variable(image)
        label = to_variable(label)
        pred = model(image)
        pred = pred.numpy()
        pred = np.squeeze(pred).astype('uint8')
        pred_shape = (pred.shape[0], pred.shape[1])
        if pred_shape[0] != prev_shape[0] or pred_shape[1] != prev_shape[1]:
            pred_map = cv2.resize(pred, prev_shape, interpolation=cv2.INTER_NEAREST)
        im_file=str(i)+'.png'
        pred_map=pred_map[:,:,0]
        # save prediction
        pred_saved_path = os.path.join(pred_saved_dir,im_file)
        mkdir(pred_saved_path)
        print(pred_saved_path)
        pred_mask = Image.fromarray(pred_map)
        pred_mask.save(pred_saved_path)




parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='/home/aistudio/work/dummy_data/')
parser.add_argument('--image_list_file', type=str, default='/home/aistudio/work/dummy_data/list.txt')
parser.add_argument('--model_dir', type=str, default='./output')
parser.add_argument('--save_dir', type=str, default='./predict')
parser.add_argument('--save_freq', type=int, default=2)

args = parser.parse_args()

# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation
    model_dir='/home/aistudio/'
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        
    # 1. create model
        if args.net == "unet":
                # TODO: create basicmodel
            model = UNet(num_classes=59)
        elif args.net == 'psp':
            model = PSPNet(num_classes=59)
        elif args.net == 'deeplab':
            model=DeepLab(num_classes=59)
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")

    # 2. load pretrained model 
        # ckpt_path = os.path.join(model_dir, 'PSPNet-Epoch-1-Loss-4.8208225727081295')
        ckpt_path = os.path.join(model_dir, 'save_model_state_dict')
        para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
        model.set_dict(para_state_dict)
        
    # 3. read test image list
        test_transforms = TrainAugmentation(256)
        basic_dataloader=BasicDataLoader(
            image_folder='/home/aistudio/work/dummy_data/',
            image_list_file=args.image_list_file,
            transform=test_transforms,
            shuffle=False
        )
        dataloader=fluid.io.DataLoader.from_generator(capacity=1,use_multiprocess=False)
        dataloader.set_sample_generator(basic_dataloader,batch_size=1,places=place)
    # 4. create transforms for test image, transform should be same as training
        
        # 5. loop over list of images

        # 6. read image and do preprocessing

        # 7. image to variable

        # 8. call inference func

        # 9. save results
        
        infer(
            model,
            model_dir=args.model_dir,
            test_dataset=dataloader,
            save_dir=args.save_dir)



if __name__ == '__main__':
    main()