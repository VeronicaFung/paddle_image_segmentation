#Bonus：
import os, sys
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
import cv2
#from unet_for_train import UNet
from deeplab import DeepLab
from pspnet import PSPNet
from unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--batch_size',type=int,default=5)
parser.add_argument('--net',type=str,default='deeplab')
parser.add_argument('--num_epochs',type=int,default=20)
parser.add_argument('--loss',type=float,default=9999.)
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--checkpoint_folder', type=str, default='/home/aistudio/work/output')
parser.add_argument('--image_folder', type=str, default='/home/aistudio/work/dummy_data')
parser.add_argument('--image_list_file', type=str, default='/home/aistudio/work/dummy_data/list.txt')

args = parser.parse_args()

def train(dataloader, model, criterion, optimizer, epoch):
    model.train()
    train_loss_meter = args.loss
    for batch_id, datas in enumerate(dataloader()):
        #TODO:
        imgs = [fluid.dygraph.to_variable(_) for _ in datas[0]]
        imgs = fluid.layers.concat(imgs,axis=0)
        labels = [fluid.dygraph.to_variable(_) for _ in datas[1]]
        labels = fluid.layers.concat(labels,axis=0)
        out = model(imgs)
        loss =  criterion(out,labels)
        loss.backward()
        optimizer.minimize(loss)
        optimizer.clear_gradients()
        args.loss=loss.numpy()[0]
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                f"Step[{batch_id:04d}], " +
                f"Average Loss: {args.loss:4f}")

    return loss

def train_dataloader(batch_size):
    '''
    数据读取
    '''
    data_dir = '/home/aistudio/work/dummy_data'
    with open(args.image_list_file,'r') as f:
        train_data = [line.strip() for line in f.readlines()]
    def reader():
        index = np.arange(len(train_data))
        mask = np.random.choice(index,batch_size,replace = False)
        imgs = []
        labels = []
        for indexs in mask:
            file_path,label_path = train_data[indexs].split()
            img_path = os.path.join(data_dir, file_path)
            label_path = os.path.join(data_dir, label_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (512, 512),cv2.INTER_NEAREST)
            # HWC to CHW
            if len(img.shape) == 3:
                img = np.transpose(img,(2,0,1))
            # 归一化
            img = np.expand_dims(img, axis=0).astype('float32')
            #read label
            label = cv2.imread(label_path).astype('float32')
            label = cv2.resize(label, (512, 512),cv2.INTER_NEAREST)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            # HWC to CHW
            if len(label.shape) == 3:
                label = np.transpose(label,(2,0,1))
            # 归一化
            label = np.expand_dims(label, axis=0)
            label = np.expand_dims(label, axis=0).astype('int64')
            assert label.shape[2:] == img.shape[2:],'ERROR'
            imgs.append(img)
            labels.append(label)

        assert len(labels) == len(mask),'ERROR'
        yield imgs,labels
    return reader

def Basic_SegLoss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape
    # TODO: create softmax_with_cross_entropy criterion
    loss = fluid.layers.softmax_with_cross_entropy(preds, labels,axis=1)
    # TODO: transpose preds to NxHxWxC
        
    mask = labels!=ignore_index
    mask = fluid.layers.cast(mask, 'float32')

    # TODO: call criterion and compute loss
    
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + 1e-5)

    return avg_loss

def main():
    # Step 0: preparation
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        #TODO: create dataloader
        train_reader = train_dataloader(args.batch_size)        
        # # Step 2: Create model
        if args.net == "unet":
            # TODO: create basicmodel
            model = UNet(num_classes=59)
        elif args.net == 'psp':
            model = PSPNet(num_classes=59)
        elif args.net == 'deeplab':
            model = DeepLab(num_classes=59)
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")


        # Step 3: Define criterion and optimizer
        criterion = Basic_SegLoss

        # create optimizer
        opt = AdamOptimizer(learning_rate=args.lr,parameter_list=model.parameters())
        # Step 4: Training
        for epoch in range(1, args.num_epochs+1):
            train_loss = train(train_reader,
                               model,
                               criterion,
                               opt,
                               epoch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {args.loss:.4f}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

                # TODO: save model and optmizer states
                fluid.dygraph.save_dygraph(model.state_dict(),'save_model_state_dict')
                fluid.dygraph.save_dygraph(opt.state_dict(),'save_opt_state_dict')


                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')



if __name__ == "__main__":
    main()