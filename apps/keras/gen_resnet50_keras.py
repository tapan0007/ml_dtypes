import re
import argparse
import tensorflow as tf
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action='store_true', help="use float16 parameters and operations")
    parser.add_argument("--dumpvars", action='store_true', help="dump variables")
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    args = parser.parse_args()

    # set Keras global configurations
    backend.set_learning_phase(0)
    if (args.fp16):
        float_type = 'float16'
        float_type2 = 'fp16'
    else:
        float_type = 'float32'
        float_type2 = 'fp32'
    backend.set_floatx(float_type)

    # load pre-trained model using Keras
    model_name = 'resnet50_%s_keras'%float_type2
    model = ResNet50(weights='imagenet')

    # various save files
    model_save_file = model_name + '.h5'
    frozen_file = model_name + '.pb'
    opt_file = model_name + '_opt.pb'

    # load image using Keras
    #img = image.load_img(args.image, target_size=(args.input_width, args.input_height))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    # run prediction
    #preds = model.predict(x)

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Keras %s prediction:'%float_type, decode_predictions(preds, top=3)[0])

    # save model in HDF5 format 
    model.save(model_save_file)

    # load model back to check that model was saved properly
    #model = keras.models.load_model(model_save_file)

    # obtain parameters
    model_input = model.input.name.replace(':0', '')
    model_output = model.output.name.replace(':0', '')
    batch, height, width, channels = model.input.shape

    print ("model, frozen file, optimized file, input size, input node, output node,")
    print ("%s, %s, %s, %dx%dx%d, %s, %s" %(model_name, frozen_file, opt_file, width, height, channels, model_input, model_output) ) 

    # obtain the TF session
    backend.set_learning_phase(0)
    sess = backend.get_session()

    # save checkpoint files for freeze_graph
    ckpt_file = '/tmp/' + model_name + '/' + model_name + '.ckpt'
    graph_file = '/tmp/' + model_name + '/' + model_name + '.pb'
    tf.train.Saver().save(sess, ckpt_file)
    tf.train.write_graph(sess.graph.as_graph_def(), logdir='.', name=graph_file, as_text=False)

    # peek at conv kernels
    if (args.dumpvars):
        # set print option precision to show that half precision has fewer signicant bits
        np.set_printoptions(precision=20)   
        for x in tf.trainable_variables():
            if (re.search("kernel", x.name)):
                print(x.name)
                print(x.dtype)
                w=sess.run(x)
                y=w.flatten()[0].item()
                print("First element (in hex): %s"%float.hex(y))
                print(w)

    # use freeze_graph to read in files and replace all variables with const
    freeze_graph(
        input_graph = graph_file,
        input_saver = "",
        input_binary = True,
        input_checkpoint = ckpt_file,
        output_node_names = model_output,
        restore_op_name = "save/restore_all",
        filename_tensor_name = "save/Const:0",
        output_graph = frozen_file,
        clear_devices = False,
        initializer_nodes = ""
        )

    # read back frozen graph
    frozen_graph_def = graph_pb2.GraphDef()
    with gfile.FastGFile(frozen_file, "rb") as f:
        frozen_graph_def.ParseFromString(f.read())
    # write out in text for debugging
    #with gfile.GFile(frozen_file+"txt", 'w') as f:
    #    f.write(text_format.MessageToString(frozen_graph_def))

    # fold batch normalization and constants and remove unused nodes 
    transformed_graph_def = graph_pb2.GraphDef()
    transformed_graph_def = TransformGraph (
             frozen_graph_def,
             [model_input],
             [model_output], 
             [
                'add_default_attributes',
                'strip_unused_nodes(type=uint8, shape="1,%d,%d,%d")'%(width, height, channels),
                'remove_nodes(op=Identity, op=CheckNumerics)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms', 
                'fold_batch_norms', 
                'fold_old_batch_norms',
                'strip_unused_nodes', 
                'sort_by_execution_order',
                'remove_attribute(attribute_name="dilations")'
             ])

    output_xformed_graph_name = model_name + "_opt.pb"
    with gfile.GFile(output_xformed_graph_name, "wb") as f:
        f.write(transformed_graph_def.SerializeToString())
    #with gfile.GFile(output_xformed_graph_name+"txt", 'w') as f:
    #    f.write(text_format.MessageToString(transformed_graph_def))


