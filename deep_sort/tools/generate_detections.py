# vim: expandtab:ts=4:sw=4

import tensorflow as tf
import os
import errno
import argparse
import numpy as np
import cv2


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    bbox = tf.cast(bbox, tf.float32)

    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox = bbox + [0., 0., (new_width - bbox[2]) / 2, 0.]
        bbox = tf.concat([bbox[:2], [new_width, bbox[3]]], axis=0)

    # convert to top left, bottom right
    bbox = tf.concat([bbox[:2], bbox[:2] + bbox[2:]], axis=0)
    bbox = tf.cast(bbox, tf.int32)

    # clip at image boundaries
    bbox = tf.concat([tf.maximum(0, bbox[:2]), bbox[2:]], axis=0)
    bbox = tf.concat([bbox[:2], tf.minimum(tf.cast(tf.shape(image)[:2][::-1] - 1, tf.int32), bbox[2:])], axis=0)
    if tf.reduce_any(bbox[:2] >= bbox[2:]):
        return None

    sx, sy, ex, ey = bbox
    image_patch = image[sy:ey, sx:ex]
    image_patch = tf.image.resize(image_patch, patch_shape[::-1])
    
    # If image_patch becomes an empty tensor, replace it with a patch of zeros
    if tf.size(image_patch) == 0:
        image_patch = tf.zeros((patch_shape[0], patch_shape[1], 3))

    return image_patch


class ImageEncoder(object):
    def __init__(self, checkpoint_filename, input_name="inception_v3_input", output_name="Identity"):
        self.session = tf.compat.v1.Session()
        with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.compat.v1.import_graph_def(graph_def, name="")
        try:
            self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(input_name + ":0")
            self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(output_name + ":0")
        except KeyError:
            layers = [i.name for i in tf.compat.v1.get_default_graph().get_operations()]
            self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(layers[0] + ':0')
            self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(layers[-1] + ':0')

        assert len(self.output_var.shape.as_list()) == 2
        assert len(self.input_var.shape.as_list()) == 4
        self.feature_dim = self.output_var.shape.as_list()[-1]
        self.image_shape = self.input_var.shape.as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

def create_box_encoder(model_filename, input_name="inception_v3_input", output_name="Identity", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    @tf.function
    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = tf.random.uniform(
                    shape=image_shape,
                    minval=0,
                    maxval=255,
                    dtype=tf.uint8
                )
            image_patches.append(patch)
        image_patches = tf.stack(image_patches)
        return image_encoder(image_patches.numpy(), batch_size)
        
    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    with tf.compat.v1.Session() as sess:
        encoder = create_box_encoder(args.model, batch_size=32)
        generate_detections(encoder, args.mot_dir, args.output_dir,
                            args.detection_dir)


if __name__ == "__main__":
    main()
