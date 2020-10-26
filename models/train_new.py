import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import argparse
from utils import AverageMeter
from basic_dataloader import BasicDataLoader
from basic_seg_loss import Basic_SegLoss
from basic_data_preprocessing import TrainAugmentation
from deeplab import DeepLab
from unet import UNet
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='deeplab')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='/home/aistudio/work/cityscapes/cityscapes/trainA/')
parser.add_argument('--label_folder', type=str, default='/home/aistudio/work/cityscapes/cityscapes/trainB/')
parser.add_argument('--image_list_file', type=str, default='/home/aistudio/work/cityscapes/cityscapes/train.txt')
parser.add_argument('--checkpoint_folder', type=str, default='/home/aistudio/work/output/')
parser.add_argument('--save_freq', type=int, default=2)

args = parser.parse_args()


def train(dataloader, model, criterion, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        # TODO:
        image = data[0]
        label = data[1]
        print(image.shape)
        print(label.shape)

        image = fluid.layers.transpose(image, (0, 3, 1, 2))
        label = fluid.layers.transpose(label, (0, 3, 1, 2))
        print(image.shape)
        print(label.shape)
        pred = model(image)

        loss = criterion(pred, label)
        #print(loss)

        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        print(loss.shape)
        # train_loss_meter.update(loss[0], n)
        train_loss_meter.update(loss.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
              f"Step[{batch_id:04d}/{total_batch:04d}], " +
              f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg

# TODO
# def valid():
        # model.eval()
        # for batch_id, data in enumerate(dataloader):
        # return avg_acc, miou


def main():
    # Step 0: preparation
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        basic_augmentation = TrainAugmentation(image_size=512)
        basic_dataloader = BasicDataLoader(image_folder=args.image_folder,
                                           label_folder=args.label_folder,
                                           image_list_file=args.image_list_file,
                                           transform=basic_augmentation,
                                           shuffle=True)
        train_dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        # valid_dataloader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=False)

        # TODO: create dataloader
        train_dataloader.set_sample_generator(basic_dataloader,
                                        batch_size=args.batch_size,
                                        places=place)
        # valid_dataloader.set_sample_generator(basic_dataloader,
        #                                       batch_size=args.batch_size,
        #                                       places=place)
        total_batch = int(len(basic_dataloader) / args.batch_size)

        # Step 2: Create model
        if args.net == "unet":
            # TODO: create basicmodel
            model = UNet(num_classes=35)
        elif args.net == 'deeplab':
            model = DeepLab(num_classes=35)
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")

        # Step 3: Define criterion and optimizer
        criterion = Basic_SegLoss

        # create optimizer
        optimizer = AdamOptimizer(learning_rate=args.lr,
                                  parameter_list=model.parameters())

        # Step 4: Training
        for epoch in range(1, args.num_epochs + 1):
            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

                # TODO
                # avg_acc, miou = valid(valid_dataloader )

                # TODO: save model and optmizer states
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)

                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')


if __name__ == "__main__":
    main()
