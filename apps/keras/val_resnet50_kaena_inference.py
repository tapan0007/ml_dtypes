import os
import glob
import shutil
import argparse
import multiprocessing
import tqdm
import numpy as np

from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend
from val_resnet50_tf_inference import load_graph, create_data_iter


def run_nn_executor(args_tuple):
    (kaena_path, image, model_path, tffe_workdir, image_path,
        rt_working_dir, output_path) = args_tuple
    if os.path.exists(output_path):
        print('INFO: found {}, skipping nn_executor'.format(output_path))
        return 0
    else:
        np.save(image_path, image)
        return os.system(
            "cd {} && "
            "{}/runtime/util/nn_executor --working_dir {} "
                "--kelf_dir . --tfpb {} --check_against_ref none "
                "--input_files {} > log-rt.txt 2>&1 ".format(
            tffe_workdir, kaena_path, rt_working_dir, model_path, image_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed; "
        "supports s3://kaena-nn-models/resnet50_fp32_keras_opt3.pb like syntax")
    parser.add_argument("--s3_profile", default='kaena', help="aws s3 profile name")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--dataset", default="1k", help="Dataset: one of 1k, 5k, or 50k")
    parser.add_argument("--stop_at_idx", type=int, default=None, help="stop processing at image index; -1 to do the entire set of images")
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")
    parser.add_argument("--kaena_path", default=os.environ.get('KAENA_PATH'), help="kaena path")
    parser.add_argument("--inkling_path", default=os.environ.get('INKLING_PATH'), help="inkling path")
    parser.add_argument("--force_tffe", action='store_true', help="always recompile the .pb file using tffe")
    parser.add_argument("--tffe_output_prefix", default="kaena_infer_", help="output file prefix used by tffe")
    parser.add_argument("--tffe_options", default="", help="tffe options")
    parser.add_argument("--parallel", type=int, default=multiprocessing.cpu_count(), help="max number of parallel nn_executor processes")
    parser.add_argument("--verbose", action='store_true', help="flag for printing inference result for each image")
    args = parser.parse_args()

    # load graph from file using TF
    graph, model_path = load_graph(args.graph, args.s3_profile)

    # get input and output operation (TF)
    input_name = "import/" + args.input_layer
    output_name = "import/" + args.output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    # set Keras global configurations
    backend.set_learning_phase(0)
    if args.fp16:
        float_type = 'float16'
    else:
        float_type = 'float32'
    backend.set_floatx(float_type)

    # create mxnet data iter
    data_iter = create_data_iter(args.dataset, args.batch_size,
        img_target_size=(args.input_width, args.input_height))

    # compile .pb into kelf
    graph_basename = os.path.splitext(os.path.basename(model_path))[0]
    tffe_workdir = os.path.abspath('kaena_work_{}'.format(graph_basename))
    kelf_path = os.path.join(tffe_workdir, 'kelf.tar.gz')
    if not os.path.exists(kelf_path) or args.force_tffe:
        if os.path.exists(tffe_workdir):
            shutil.rmtree(tffe_workdir)
        trial_image = data_iter.next().data[0].asnumpy()
        trial_image = np.transpose(trial_image, (0, 2, 3, 1)) # NCHW -> NHWC
        trial_image = preprocess_input(trial_image)
        os.makedirs(tffe_workdir, exist_ok=True)
        trial_image_path = os.path.join(tffe_workdir, 'trial_image.npy')
        np.save(trial_image_path, trial_image)
        print('INFO: running tffe to compile {}'.format(model_path),
            flush=True)
        status = os.system(
            "cd {} && "
            "{}/compiler/scripts/tffe --out_prefix {} --parallel_streams "
                "--tfpb {} --input_node {} --focus_to {} "
                "--images {} --batch {} {} > log-fe.txt 2>&1 ".format(
            tffe_workdir, args.kaena_path, args.tffe_output_prefix,
            model_path, args.input_layer, args.output_layer,
            trial_image_path, args.batch_size, args.tffe_options))
        if status:
            print('ERROR: tffe failed with status {:d}'.format(status))
            exit(1)
        data_iter.reset()
        print('INFO: finished tffe compilation', flush=True)

    os.system(
        "cd {} && ln -sf {} cached_tf.pb".format(tffe_workdir, model_path))
    batch_list = []
    batch_args_list = []
    print('INFO: preparing batch inputs...')
    cnt = 0
    for bidx, batch in enumerate(data_iter):
        cnt += args.batch_size - batch.pad
        if args.stop_at_idx is not None and cnt >= args.stop_at_idx:
            break
        image = batch.data[0].asnumpy()
        image = np.transpose(image, (0, 2, 3, 1)) # NCHW -> NHWC
        image = preprocess_input(image)
        image_name = 'input_batch_{:d}.npy'.format(bidx)
        image_path = os.path.join(tffe_workdir, image_name)
        output_name = args.tffe_output_prefix + \
            args.output_layer.replace('/', '__') + ':0-out.npy'
        rt_working_dir = 'working_dir_batch_{:d}'.format(bidx)
        output_path = os.path.join(tffe_workdir, rt_working_dir, output_name)
        output_path = os.path.abspath(output_path)
        batch_list.append((output_path, batch.label[0].asnumpy(), batch.pad))
        batch_args_list.append((
            args.kaena_path, image, model_path, tffe_workdir, image_path,
            rt_working_dir, output_path))
    print('INFO: running {:d} nn_executor processes...'.format(args.parallel))
    with multiprocessing.Pool(args.parallel) as pool:
        pool_imap = pool.imap_unordered(run_nn_executor, batch_args_list)
        for _ in tqdm.tqdm(pool_imap, total=len(batch_args_list)):
            pass
    results_labels_list = []
    for bidx, (output_path, labels, pad) in enumerate(batch_list):
        results = np.load(output_path)
        results = results[:len(results)-pad]
        results_labels_list.append((results, labels))

    top5 = 0
    top1 = 0
    cnt = 0
    for results, labels in results_labels_list:
        # take the top 5
        for result, label in zip(results, labels):
            cnt += 1
            top_indices = result.argsort()[-5:][::-1]
            if (len([i for i in top_indices if i==label])>0):
                top5 += 1
            if (label == top_indices[0]):
                top1 += 1

            # decode the results into a list of tuples (class, description, probability) using Keras
            if args.verbose:
                print("\n---------- %s image %d label %d ----------"%(os.path.basename(args.graph), cnt, label))
                print('TF    %s prediction: '%float_type, decode_predictions(result[np.newaxis, ...], top=3)[0])
    print('Accuracy scores: cnt %d top1 %d top5 %d top1 %f top5 %f'%(cnt, top1, top5, top1/cnt, top5/cnt))
