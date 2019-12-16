import tensorflow as tf
import tensorflow_datasets as tfds
import dataIO
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import csv
from datetime import datetime


padding = "SAME"
batch_size = 8
learning_rate = 0.001
image_size = 352
epochs = 5

#Mouse
seg_filepath = "/home/frtim/Documents/Code/SomaeDetection/Mouse/seg_Mouse_773x832x832.h5"
somae_filepath = "/home/frtim/Documents/Code/SomaeDetection/Mouse/somae_reduced_cut_Mouse_773x832x832.h5"

seg_data = dataIO.ReadH5File(seg_filepath, [1])
somae_raw = dataIO.ReadH5File(somae_filepath, [1])

z_max = min(seg_data.shape[0],somae_raw.shape[0])

somae_data = np.zeros((z_max,seg_data.shape[1],seg_data.shape[2]),dtype=np.uint64)
somae_data[:,:somae_raw.shape[1],:somae_raw.shape[2]]=somae_raw[:z_max,:,:]

seg_data = seg_data[:,:,:z_max]

seg_data[seg_data>0]=1
somae_data[somae_data>0]=1

seg_data = seg_data[:,::2,::2]
somae_data = somae_data[:,::2,::2]

seg_data = seg_data[:,:image_size,:image_size]
somae_data = somae_data[:,:image_size,:image_size]

# find maximum z coordinate
val_data_size = 64

data_in = np.zeros((seg_data.shape[0],seg_data.shape[1],seg_data.shape[2],2), dtype=np.uint8)

data_in[:,:,:,0] = seg_data
data_in[:,:,:,1] = somae_data

validation_data = data_in[:val_data_size,:,:,:]
train_data = data_in[val_data_size:,:,:,:]

class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss form logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
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

def conv2d(inputs , filters):
    out = tf.nn.conv2d( inputs , filters , strides=1 , padding=padding )
    return tf.nn.relu(out)

def conv2d_T(inputs , filters):
    strides = 2
    output_shape_ = inputs.shape.as_list()
    output_shape_[1]=output_shape_[1]*strides
    output_shape_[2]=output_shape_[2]*strides
    output_shape_[3]=filters.shape[2]
    out = tf.nn.conv2d_transpose( inputs , filters, output_shape=output_shape_, strides=2 , padding=padding )
    return tf.nn.relu(out)

def maxpool2(inputs):
    return tf.nn.max_pool2d( inputs , ksize=2 , padding=padding , strides=2 )

def maxpool4(inputs):
    return tf.nn.max_pool2d( inputs , ksize=4 , padding=padding , strides=4 )

# filters = [1,64,64,128,128,256,256,512,512,1024,1024,512,512,512,256,256,256,128,128,128,64,64,64,1]
# filters = [1,64,128,256,512,1024]
filters = [1,16,32,54,128,256]

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
    [ 3, 3, filters[5],           filters[5]],     #L52 -> L53

    [ 2, 2, filters[4],           filters[5]],     #L53 -> L44
    [ 3, 3, 2*filters[4],         filters[4]],     #L44 -> L45
    [ 3, 3, filters[4],           filters[4]],     #L45 -> L46
    [ 2, 2, filters[3],           filters[4]],     #L46 -> L34
    [ 3, 3, 2*filters[3],         filters[3]],     #L34 -> L35
    [ 3, 3, filters[3],           filters[3]],     #L35 -> L36
    [ 2, 2, filters[2],           filters[3]],     #L36 -> L24
    [ 3, 3, 2*filters[2],         filters[2]],     #L24 -> L25
    [ 3, 3, filters[2],           filters[2]],     #L25 -> L26
    [ 2, 2, filters[1],           filters[2]],     #L25 -> L14
    [ 3, 3, 2*filters[1],           filters[1]],     #L14 -> L15
    [ 3, 3, filters[1],           filters[1]],     #L15 -> L16

    [ 1, 1, filters[1],           filters[0]],     #L16 -> L17
]

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

    def restoreWeights(self, checkpoint_directory):
        status = self.ckpt.restore(tf.train.latest_checkpoint(checkpoint_directory))
        status.assert_consumed()  # Optional check

weights = model_weights(shapes)
# weights.restoreWeights()

@tf.function
def model( L11 ) :
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

w_loss = WeightedBinaryCrossEntropy(13, 1)

def loss( gt , pred ):
    return w_loss( gt , pred )

optimizer = tf.optimizers.Adam( learning_rate )
train_acc = tf.metrics.BinaryAccuracy()
valid_acc = tf.metrics.BinaryAccuracy()

def train_step( model, inputs , gt ):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        current_loss = loss( gt, pred)
    grads = tape.gradient( current_loss , weights.values )
    optimizer.apply_gradients( zip( grads , weights.values ) )
    train_loss = tf.reduce_mean( current_loss )
    train_acc.update_state(tf.reshape(gt,[-1]), tf.reshape(pred,[-1]))

    return train_loss

def predict_step( model, inputs, gt):
    pred = model(inputs)
    current_loss = loss( gt, pred)
    val_loss = tf.reduce_mean( current_loss )
    valid_acc.update_state(tf.reshape(gt,[-1]), tf.reshape(pred,[-1]))

    return pred, val_loss

train_ids = np.random.permutation(train_data.shape[0])
valid_ids = np.random.permutation(validation_data.shape[0])

for e in range( epochs ):
    print("Epoch: " + str(e))
    count = 0
    train_loss = 0
    for k in train_ids:

        if count%50 == 0 and count>0:
            print(str(count) + "/" + str(len(train_ids)) )
            print("Train loss: " + str(train_loss.numpy()/count))
            print("Train accu: " + str(train_acc.result().numpy()))

        image = train_data[k:k+batch_size,:,:,0,None]
        mask = train_data[k:k+batch_size,:,:,1,None]

        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask_gt = tf.convert_to_tensor( mask , dtype=tf.float32 )

        curr_loss = train_step( model , image , mask_gt )
        train_loss += curr_loss

        count += 1

    val_loss_best = 1000000000
    val_loss = 0
    for j in valid_ids:
        image = validation_data[None, j,:,:,0,None]
        mask = validation_data[None,j,:,:,1,None]
        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask_gt = tf.convert_to_tensor( mask , dtype=tf.float32 )
        _, cur_loss = predict_step(model , image, mask_gt)
        val_loss+=cur_loss

    if val_loss<val_loss_best:
        val_loss_best = val_loss
        weights.saveWeights()
        print("Weights saved ------------------")

    val_loss = val_loss.numpy()/len(valid_ids)
    print("Valid loss: " + str(val_loss))
    print("Valid accu: " + str(valid_acc.result().numpy()))

    train_acc.reset_states()
    valid_acc.reset_states()

# for j in valid_ids[::5]:
#
#     image = validation_data[None, j,:,:,0,None]
#     mask = validation_data[None,j,:,:,1,None]
#     image = tf.convert_to_tensor( image , dtype=tf.float32 )
#     mask_gt = tf.convert_to_tensor( mask , dtype=tf.float32 )
#     mask_pred, _ = predict_step(model , image, mask_gt)
#
#     fig = plt.figure(figsize=(20, 12))
#     fig.subplots_adjust(hspace=0.4, wspace=0.4)
#     ax = fig.add_subplot(1, 3, 1)
#     ax.imshow(np.reshape(image[0,:,:,0], (image_size, image_size)), cmap="gray")
#     ax = fig.add_subplot(1, 3, 2)
#     ax.imshow(np.reshape(mask_gt[0,:,:,0]*255, (image_size, image_size)), cmap="gray")
#     ax = fig.add_subplot(1, 3, 3)
#     ax.imshow(np.reshape(mask_pred[0,:,:,0]*255, (image_size, image_size)), cmap="gray")
#     plt.show()
