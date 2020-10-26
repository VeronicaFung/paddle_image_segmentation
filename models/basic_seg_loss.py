import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

def Basic_SegLoss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape
    # TODO: create softmax_with_cross_entropy criterion
    # TODO: transpose preds to NxHxWxC
    preds = fluid.layers.transpose(preds, [0, 2, 3, 1])
    preds = fluid.layers.reshape(preds, [-1, c])
    labels = fluid.layers.reshape(labels, [-1, 1])
    labels = fluid.layers.cast(labels, 'int64')
    mask = labels!=ignore_index
    mask = fluid.layers.cast(mask, 'float32')
    # TODO: call criterion and compute loss
    loss = fluid.layers.softmax_with_cross_entropy(preds,labels,ignore_index=ignore_index)
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)

    return avg_loss

def main():
    label = cv2.imread('./dummy_data/GroundTruth_trainval_png/2008_000002.png',0).astype(np.int64)
    pred = np.random.uniform(0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[:,:,np.newaxis].transpose((2, 0, 1))
    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = Basic_SegLoss(pred, label)
        print(loss)

if __name__ == "__main__":
    main()