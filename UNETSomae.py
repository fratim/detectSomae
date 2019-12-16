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

filters = [1,64,64,128,128,256,256,512,512,1024,1024,512,512,512,256,256,256,128,128,128,64,64,64,1]

shapes = [
    [ 3, 3, filters[0],           filters[1]],      #L11 -> L12
    [ 3, 3, filters[1],           filters[2]],      #L12 -> L13
    [ 3, 3, filters[2],           filters[3]],      #L21 -> L22
    [ 3, 3, filters[3],           filters[4]],      #L22 -> L23
    [ 3, 3, filters[4],           filters[5]],      #L31 -> L32
    [ 3, 3, filters[5],           filters[6]],      #L32 -> L33
    [ 3, 3, filters[6],           filters[7]],      #L41 -> L42
    [ 3, 3, filters[7],           filters[8]],      #L42 -> L43
    [ 3, 3, filters[8],           filters[9]],      #L51 -> L52
    [ 3, 3, filters[9],           filters[10]],     #L52 -> L53

    [ 2, 2, filters[11],           filters[10]],     #L53 -> L44
    [ 3, 3, filters[11],           filters[12]],     #L44 -> L45
    [ 3, 3, filters[12],           filters[13]],     #L45 -> L46
    [ 2, 2, filters[14],           filters[13]],     #L46 -> L34
    [ 3, 3, filters[14],           filters[15]],     #L34 -> L35
    [ 3, 3, filters[15],           filters[16]],     #L35 -> L36
    [ 2, 2, filters[17],           filters[16]],     #L36 -> L24
    [ 3, 3, filters[17],           filters[18]],     #L24 -> L25
    [ 3, 3, filters[18],           filters[19]],     #L25 -> L26
    [ 2, 2, filters[20],           filters[19]],     #L25 -> L14
    [ 3, 3, filters[20],           filters[21]],     #L14 -> L15
    [ 3, 3, filters[21],           filters[22]],     #L15 -> L16

    [ 1, 1, filters[22],           filters[23]],     #L16 -> L17
]

weights = []
for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )

def model( L11 ) :
    L12 = conv2d(L11, weights[0])
    L13 = conv2d(L12, weights[1])

    L21 = maxpool2(L13)
    L22 = conv2d(L21, weights[2])
    L23 = conv2d(L22, weights[3])

    L31 = maxpool2(L23)
    L32 = conv2d(L31, weights[4])
    L33 = conv2d(L32, weights[5])

    L41 = maxpool2(L33)
    L42 = conv2d(L41, weights[6])
    L43 = conv2d(L42, weights[7])

    L51 = maxpool2(L43)
    L52 = conv2d(L51, weights[8])
    L53 = conv2d(L52, weights[9])

    L44 = conv2d_T(L53, weights[10])
    # L44 = tf.concat([L43,L44_],axis=3)
    L45 = conv2d(L44, weights[11])
    L46 = conv2d(L45, weights[12])

    L34 = conv2d_T(L46, weights[13])
    L35 = conv2d(L34, weights[14])
    L36 = conv2d(L35, weights[15])

    L24 = conv2d_T(L36, weights[16])
    L25 = conv2d(L24, weights[17])
    L26 = conv2d(L25, weights[18])

    L14 = conv2d_T(L26, weights[19])
    L15 = conv2d(L14, weights[20])
    L16 = conv2d(L15, weights[21])

    L17 = conv2d(L16, weights[22])

    return tf.nn.softmax(L17)


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

    for j in range(10):

        print(j)

        image = validation_data[None, j,:,:,0,None]
        mask = validation_data[None,j,:,:,1,None]

        image = tf.convert_to_tensor( image , dtype=tf.float32 )
        mask_gt = tf.convert_to_tensor( mask , dtype=tf.float32 )

        mask_pred = predict_step( model , image)

        fig = plt.figure(figsize=(20, 12))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(np.reshape(image[0,:,:,0], (image_size, image_size)), cmap="gray")
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(np.reshape(mask_gt[0,:,:,0]*255, (image_size, image_size)), cmap="gray")
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(np.reshape(mask_pred[0,:,:,0]*255, (image_size, image_size)), cmap="gray")
        plt.show()
