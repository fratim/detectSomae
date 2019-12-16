import tensorflow as tf
import tensorflow_datasets as tfds
import dataIO
import matplotlib.pyplot as plt
import numpy as np

padding = "SAME"
batch_size = 1
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

print("data_in shape: "  + str(data_in.shape))

data_in[:,:,:,0] = seg_data
data_in[:,:,:,1] = somae_data

validation_data = data_in[:val_data_size,:,:,:]
train_data = data_in[val_data_size:,:,:,:]

def conv2d(inputs , filters):
    out = tf.nn.conv2d( inputs , filters , strides=1 , padding=padding )
    return tf.nn.relu(out)

def conv2d_T(inputs , filters):
    strides = 2
    output_shape_ = inputs.shape.as_list()
    output_shape_[1]=output_shape_[1]*strides
    output_shape_[2]=output_shape_[2]*strides
    out = tf.nn.conv2d_transpose( inputs , filters, output_shape=output_shape_, strides=2 , padding=padding )
    return tf.nn.relu(out)

def maxpool2(inputs):
    return tf.nn.max_pool2d( inputs , ksize=2 , padding=padding , strides=2 )

def maxpool4(inputs):
    return tf.nn.max_pool2d( inputs , ksize=4 , padding=padding , strides=4 )



output_classes = 1
initializer = tf.initializers.RandomNormal()
def get_weight( shape , name ):
    return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

filters = [4,8,8,4]

shapes = [
    [ 3, 3, 1,           filters[0]],      #input -> c1

    [ 3, 3, filters[0],  filters[1]],      #p1 -> c2

    [ 3, 3, filters[1],  filters[2]],      #c2 -> u1

    [ 3, 3, filters[2],  filters[3]],      #u1 -> c3

    [ 1, 1, filters[3],  output_classes ]  #c3 -> outp
]

weights = []
for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )

def model( x ) :
    print("x:" + str(x.shape))
    c1 = conv2d(x, weights[0])
    print("c1:" + str(c1.shape))
    p1 = maxpool2(c1)
    print("p1:" + str(p1.shape))
    c2 = conv2d(p1, weights[1])
    print("c2:" + str(c2.shape))
    u1 = conv2d_T(c2, weights[2])
    print("u1:" + str(u1.shape))
    c3 = conv2d(u1, weights[3])
    print("c3:" + str(c3.shape))
    output = conv2d(c3, weights[4])
    print("output:" + str(output.shape))
    return tf.nn.softmax( output )


def loss( pred , target ):
    return tf.losses.categorical_crossentropy( target , pred )

optimizer = tf.optimizers.Adam( learning_rate )

def train_step( model, inputs , outputs ):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        current_loss = loss( pred, outputs)
    grads = tape.gradient( current_loss , weights )
    optimizer.apply_gradients( zip( grads , weights ) )
    print( tf.reduce_mean( current_loss ) )

def predict_step( model, inputs):
    pred = model(inputs)
    return pred

for e in range( epochs ):
    print("Epoch: " + str(e))
    for k in range(train_data.shape[0]):
        print(k)

        image = train_data[None, k,:,:,0,None]
        mask = train_data[None,k,:,:,1,None]

        print(image.shape)
        print(mask.shape)
        # fig = plt.figure(figsize=(20, 12))
        # fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # ax = fig.add_subplot(1, 2, 1)
        # ax.imshow(np.reshape(image[0,:,:,0], (image_size, image_size)), cmap="gray")
        # ax = fig.add_subplot(1, 2, 2)
        # ax.imshow(np.reshape(mask[0,:,:,0]*255, (image_size, image_size)), cmap="gray")
        # plt.show()

        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask = tf.convert_to_tensor( mask , dtype=tf.float32 )

        train_step( model , image , mask )

    for j in range(validation_data.shape[0]):
        print(j)
        image = validation_data[None, k,:,:,0,None]
        mask = validation_data[None,k,:,:,1,None]

        # fig = plt.figure(figsize=(20, 12))
        # fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # ax = fig.add_subplot(1, 2, 1)
        # ax.imshow(np.reshape(image[0,:,:,0], (image_size, image_size)), cmap="gray")
        # ax = fig.add_subplot(1, 2, 2)
        # ax.imshow(np.reshape(mask[0,:,:,0]*255, (image_size, image_size)), cmap="gray")
        # plt.show()

        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask = tf.convert_to_tensor( mask , dtype=tf.float32 )

        predict_step( model , image , mask )
