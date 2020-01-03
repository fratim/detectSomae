import tensorflow as tf
import tensorflow_datasets as tfds
import dataIO
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime

padding = "SAME"
batch_size = 8
depth = 4
learning_rate = 0.001
image_size = 704
epochs = 100

# specify size of validation dataset
val_data_size = 64

# preprocess input data
def prepareDataTraining(seg_data, somae_data):

    # mask the data to be binary
    seg_data[seg_data>0]=1
    somae_data[somae_data>0]=1

    #downsample in x and y direction
    seg_data = seg_data[:,:,:]
    somae_data = somae_data[:,:,:]

    # cut to image size of Zebrafinch data
    seg_data = seg_data[:,:image_size,:image_size]
    somae_data = somae_data[:,:image_size,:image_size]

    # create object to hold elements for 3D input tensors of depth(*2)+1
    seg_deep = np.zeros((seg_data.shape[0],seg_data.shape[1],seg_data.shape[2],depth*2+1), dtype=np.uint8)

    # populate deep segmentation tensor
    seg_deep[:,:,:,depth]=seg_data
    for d in range(1,depth+1):
        seg_deep[:-d,:,:,depth+d]=seg_data[d:,:,:]
        seg_deep[d:,:,:,depth-d]=seg_data[:-d,:,:]

    # cut training and validationd ataset
    valid_seg = seg_deep[:val_data_size,:,:,:]
    valid_mask = somae_data[:val_data_size,:,:]
    train_seg = seg_deep[val_data_size:,:,:,:]
    train_mask = somae_data[val_data_size:,:,:]

    # shuffle both training and validation data
    valid_ids = np.random.permutation(valid_seg.shape[0])
    train_ids = np.random.permutation(train_seg.shape[0])

    valid_seg[:,:,:] = valid_seg[valid_ids,:,:,:]
    valid_mask[:,:,:] = valid_mask[valid_ids,:,:]
    train_seg[:,:,:] = train_seg[train_ids,:,:,:]
    train_mask[:,:,:] = train_mask[train_ids,:,:]

    return train_seg, train_mask, valid_seg, valid_mask

# preprocess input data
def prepareDataPrediction(seg_data):

    # mask the data to be binary
    seg_data[seg_data>0]=1

    #downsample in x and y direction
    seg_data = seg_data[:,:,:]

    # cut to image size of Zebrafinch data
    seg_data = seg_data[:,:image_size,:image_size]

    # create object to hold elements for 3D input tensors of depth(*2)+1
    seg_deep = np.zeros((seg_data.shape[0],seg_data.shape[1],seg_data.shape[2],depth*2+1), dtype=np.uint8)

    # populate deep segmentation tensor
    seg_deep[:,:,:,depth]=seg_data
    for d in range(1,depth+1):
        seg_deep[:-d,:,:,depth+d]=seg_data[d:,:,:]
        seg_deep[d:,:,:,depth-d]=seg_data[:-d,:,:]

    # cut training and validationd ataset
    valid_seg = seg_deep[:,:,:,:]

    return valid_seg

# define the weighted loss function
class WeightedBinaryCrossEntropy(tf.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss form logits or the probability.
      reduction: Type of tf.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=tf.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction,
                                                         name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            # Manually calculate the weighted cross entropy.
            # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            # where z are labels, x is logits, and q is the weight.
            # Since the values passed are from sigmoid (assuming in this case)
            # sigmoid(x) will be replaced by y_pred

            # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
            x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)

            # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
            x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6)

            return tf.add(x_1, x_2) * self.weight

        # Use built in function
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight

# convolutional layer
def conv2d(inputs , filters):
    out = tf.nn.conv2d( inputs , filters , strides=1 , padding=padding )
    return tf.nn.relu(out)

# transposed convolutional layer
def conv2d_T(inputs , filters):
    strides = 2
    output_shape_ = inputs.shape.as_list()
    output_shape_[1]=output_shape_[1]*strides
    output_shape_[2]=output_shape_[2]*strides
    output_shape_[3]=filters.shape[2]
    out = tf.nn.conv2d_transpose( inputs , filters, output_shape=output_shape_, strides=2 , padding=padding )
    return tf.nn.relu(out)

# max pool layer of size 2
def maxpool2(inputs):
    return tf.nn.max_pool2d( inputs , ksize=2 , padding=padding , strides=2 )

# max pool layer of size 4
def maxpool4(inputs):
    return tf.nn.max_pool2d( inputs , ksize=4 , padding=padding , strides=4 )

# model weights
class model_weights:
    def __init__(self, shapes):
        self.values = []
        self.checkpoint_path = './ckpt_'+ datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
        initializer = tf.initializers.RandomNormal()
        def get_weight( shape , name ):
            return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

        for i in range( len( shapes ) ):
            self.values.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )

        self.ckpt = tf.train.Checkpoint(**{f'values{i}': v for i, v in enumerate(self.values)})

    def saveWeights(self):
        self.ckpt.save(self.checkpoint_path)

    def restoreWeights(self, ckpt_restore):
        print("restoring weights from: " + str(ckpt_restore))
        status = self.ckpt.restore(ckpt_restore)
        status.assert_consumed()  # Optional check

#define UNET model
@tf.function #TODO change this back
def model(L11, weights) :
    L12 = conv2d(L11, weights.values[0])
    L13 = conv2d(L12, weights.values[1])

    L21 = maxpool2(L13)
    L22 = conv2d(L21, weights.values[2])
    L23 = conv2d(L22, weights.values[3])

    L31 = maxpool2(L23)
    L32 = conv2d(L31, weights.values[4])
    L33 = conv2d(L32, weights.values[5])

    L41 = maxpool2(L33)
    L42 = conv2d(L41, weights.values[6])
    L43 = conv2d(L42, weights.values[7])

    L51 = maxpool2(L43)
    L52 = conv2d(L51, weights.values[8])
    L53 = conv2d(L52, weights.values[9])

    L44_ = conv2d_T(L53, weights.values[10])
    L44 = tf.concat([L43,L44_],axis=3)
    L45 = conv2d(L44, weights.values[11])
    L46 = conv2d(L45, weights.values[12])

    L34_ = conv2d_T(L46, weights.values[13])
    L34 = tf.concat([L33,L34_],axis=3)
    L35 = conv2d(L34, weights.values[14])
    L36 = conv2d(L35, weights.values[15])

    L24_ = conv2d_T(L36, weights.values[16])
    L24 = tf.concat([L23,L24_],axis=3)
    L25 = conv2d(L24, weights.values[17])
    L26 = conv2d(L25, weights.values[18])

    L14_ = conv2d_T(L26, weights.values[19])
    L14 = tf.concat([L13,L14_],axis=3)
    L15 = conv2d(L14, weights.values[20])
    L16 = conv2d(L15, weights.values[21])

    L17 = conv2d(L16, weights.values[22])

    return tf.nn.sigmoid(L17)

def initializeModel(restore, ckpt_restore):
    # filters for the UNEt layers:
    # filters = [depth*2+1,64,128,256,512,1024,1]   #original UNET
    filters = [depth*2+1,16,32,54,128,256,1]        # modified, lighter UNET

    #shapes of the weight tensors
    shapes = [
        [ 3, 3, filters[0],           filters[1]],      #L11 -> L12
        [ 3, 3, filters[1],           filters[1]],      #L12 -> L13
        [ 3, 3, filters[1],           filters[2]],      #L21 -> L22
        [ 3, 3, filters[2],           filters[2]],      #L22 -> L23
        [ 3, 3, filters[2],           filters[3]],      #L31 -> L32
        [ 3, 3, filters[3],           filters[3]],      #L32 -> L33
        [ 3, 3, filters[3],           filters[4]],      #L41 -> L42
        [ 3, 3, filters[4],           filters[4]],      #L42 -> L43
        [ 3, 3, filters[4],           filters[5]],      #L51 -> L52
        [ 3, 3, filters[5],           filters[5]],      #L52 -> L53

        [ 2, 2, filters[4],           filters[5]],      #L53 -> L44
        [ 3, 3, 2*filters[4],         filters[4]],      #L44 -> L45
        [ 3, 3, filters[4],           filters[4]],      #L45 -> L46
        [ 2, 2, filters[3],           filters[4]],      #L46 -> L34
        [ 3, 3, 2*filters[3],         filters[3]],      #L34 -> L35
        [ 3, 3, filters[3],           filters[3]],      #L35 -> L36
        [ 2, 2, filters[2],           filters[3]],      #L36 -> L24
        [ 3, 3, 2*filters[2],         filters[2]],      #L24 -> L25
        [ 3, 3, filters[2],           filters[2]],      #L25 -> L26
        [ 2, 2, filters[1],           filters[2]],      #L25 -> L14
        [ 3, 3, 2*filters[1],         filters[1]],      #L14 -> L15
        [ 3, 3, filters[1],           filters[1]],      #L15 -> L16

        [ 1, 1, filters[1],           filters[6]],      #L16 -> L17
    ]

    weights = model_weights(shapes)

    if restore:
        weights.restoreWeights(ckpt_restore)

    # initialize loss
    w_loss = WeightedBinaryCrossEntropy(13, 1)

    # initialize optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    # initialize accuracy objects
    train_acc = tf.metrics.BinaryAccuracy()
    valid_acc = tf.metrics.BinaryAccuracy()
    train_loss = tf.metrics.Mean()
    valid_loss = tf.metrics.Mean()

    return weights, w_loss, optimizer, train_acc, valid_acc, train_loss, valid_loss

# define train step
def train_step(model, weights, inputs, gt, optimizer, w_loss, train_loss, train_acc):
    with tf.GradientTape() as tape:
        pred = model(inputs, weights)
        current_loss = w_loss( gt, pred)
    grads = tape.gradient(current_loss, weights.values )
    optimizer.apply_gradients(zip(grads , weights.values ) )
    train_loss.update_state(current_loss)
    train_acc.update_state(gt, pred)

    return optimizer

#define prediction step
def predict_step(model, weights, inputs, gt, w_loss, valid_loss, valid_acc): #TODO remove paqssing of model here
    pred = model(inputs, weights)
    current_loss = w_loss( gt, pred)
    valid_loss.update_state(current_loss)
    valid_acc.update_state(gt, pred)

    return pred


def trainOnEpochs(train_seg, train_mask, valid_seg, valid_mask, weights, w_loss, optimizer, train_acc, valid_acc, train_loss, valid_loss):

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    valid_loss_best = 1000000000
    for epoch in range(epochs):

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', valid_acc.result(), step=epoch)
        train_acc.reset_states()
        valid_acc.reset_states()
        train_loss.reset_states()
        valid_loss.reset_states()
        print("---------------------")
        print("Epoch: " + str(epoch))

        for k in np.arange(0,train_seg.shape[0],batch_size):

            image = train_seg[k:k+batch_size,:,:,:]
            mask = train_mask[k:k+batch_size,:,:,None]
            image = tf.convert_to_tensor(image, dtype=tf.float32 )
            mask_gt = tf.convert_to_tensor(mask, dtype=tf.float32 )
            optimizer = train_step(model, weights, image, mask_gt, optimizer, w_loss, train_loss, train_acc)

        for j in np.arange(0,valid_seg.shape[0],batch_size):

            image = valid_seg[j:j+batch_size,:,:,:]
            mask = valid_mask[j:j+batch_size,:,:,None]
            image = tf.convert_to_tensor( image , dtype=tf.float32 )
            mask_gt = tf.convert_to_tensor( mask , dtype=tf.float32 )
            mask_pred = predict_step(model, weights, image, mask_gt, w_loss, valid_loss, valid_acc).numpy()
            mask_pred_thresh = mask_pred
            mask_pred_thresh[mask_pred_thresh<=0.5]=0
            mask_pred_thresh[mask_pred_thresh>0.5]=1
            mask_pred_thresh = tf.convert_to_tensor(mask_pred_thresh, np.float32)

            if epoch%5==0:
                with valid_summary_writer.as_default():
                    tf.summary.image("valid-epoch"+str(epoch)+"j-"+str(j), tf.concat([tf.expand_dims(image[:,:,:,depth],3), mask_gt, mask_pred_thresh],axis=1), step=epoch, max_outputs=5)

        # if valid_loss.result().numpy()<valid_loss_best:
        #     valid_loss_best = valid_loss.result().numpy()
        #     weights.saveWeights()
        #     print("Weights saved ------------------")

        print("Train loss: " + str(train_loss.result().numpy()))
        print("Train accu: " + str(train_acc.result().numpy()))
        print("Valid loss: " + str(valid_loss.result().numpy()))
        print("Valid accu: " + str(valid_acc.result().numpy()))

        weights.saveWeights()
        print("Weights saved ------------------")


def TrainOnMouse(restore, ckpt_restore):

    # Mouse
    seg_filepath =      "/home/frtim/Documents/Code/SomaeDetection/Mouse/gt_data/seg_Mouse_762x832x832.h5"
    somae_filepath =    "/home/frtim/Documents/Code/SomaeDetection/Mouse/gt_data/somae_Mouse_762x832x832.h5"

    seg_data = dataIO.ReadH5File(seg_filepath, [1])
    somae_data = dataIO.ReadH5File(somae_filepath, [1])

    train_seg, train_mask, valid_seg, valid_mask = prepareDataTraining(seg_data, somae_data)

    weights, w_loss, optimizer, train_acc, valid_acc, train_loss, valid_loss = initializeModel(restore=restore, ckpt_restore=ckpt_restore)

    trainOnEpochs(train_seg, train_mask, valid_seg, valid_mask, weights, w_loss, optimizer, train_acc, valid_acc, train_loss, valid_loss)

def PredictOnZebrafinch(ckpt_restore):

    # Zebrafinch
    seg_filepath =      "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/Zebrafinch-seg-dsp_8.h5"
    seg_data = dataIO.ReadH5File(seg_filepath, [1])

    valid_seg = prepareDataPrediction(seg_data)

    weights, w_loss, optimizer, train_acc, valid_acc, train_loss, valid_loss = initializeModel(restore=True, ckpt_restore=ckpt_restore)

    somae_mask_out = np.zeros((valid_seg.shape[0],valid_seg.shape[1],valid_seg.shape[2]), dtype=np.float64)
    print("Output shape: " + str(somae_mask_out.shape))

    for j in np.arange(0,valid_seg.shape[0],batch_size):

        image = valid_seg[j:j+batch_size,:,:,:]
        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask_pred = tf.squeeze(model(image, weights))

        somae_mask_out[j:j+batch_size,:,:]=mask_pred.numpy()

        print(str(j).zfill(4))

    somae_mask_out[somae_mask_out<=0.5]=0
    somae_mask_out[somae_mask_out>0.5]=1

    somae_mask_out = somae_mask_out.astype(np.uint64)

    dataIO.WriteH5File(somae_mask_out, "/home/frtim/Documents/Code/SomaeDetection/Zebrafinch/Zebrafinch-somae_new-dsp_8.h5","main")

def PredictOnMouse(ckpt_restore):
    # Zebrafinch
    seg_filepath =      "/home/frtim/Documents/Code/SomaeDetection/Mouse/gt_data/seg_Mouse_762x832x832.h5"
    seg_data = dataIO.ReadH5File(seg_filepath, [1])

    valid_seg = prepareDataPrediction(seg_data)

    weights, w_loss, optimizer, train_acc, valid_acc, train_loss, valid_loss = initializeModel(restore=True, ckpt_restore=ckpt_restore)

    somae_mask_out = np.zeros((valid_seg.shape[0],valid_seg.shape[1],valid_seg.shape[2]), dtype=np.float64)
    print("Output shape: " + str(somae_mask_out.shape))

    for j in np.arange(0,valid_seg.shape[0],batch_size):

        image = valid_seg[j:j+batch_size,:,:,:]
        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask_pred = tf.squeeze(model(image, weights))

        somae_mask_out[j:j+batch_size,:,:]=mask_pred.numpy()

        print(str(j).zfill(4))

    somae_mask_out[somae_mask_out<=0.5]=0
    somae_mask_out[somae_mask_out>0.5]=1

    somae_mask_out = somae_mask_out.astype(np.uint64)

    dataIO.WriteH5File(somae_mask_out, "/home/frtim/Documents/Code/SomaeDetection/Mouse/gt_data/somae_pred_Mouse.h5","main")


def main():
    # restore from checkpoint
    ckpt_restore = '/home/frtim/Documents/Code/SomaeDetection/ckpt_20200103-092925/-90'

    # TrainOnMouse(restore=True, ckpt_restore=ckpt_restore)
    # PredictOnZebrafinch(ckpt_restore)
    PredictOnMouse(ckpt_restore)


if True == 1:
    main()
