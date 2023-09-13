# python video_write.py /home/lyn/Videos/flare1.mp4 /home/lyn/HitLyn/metrabs/outputs/flare1.txt
import sys
import numpy as np

import tensorflow as tf
import cv2
from IPython import embed

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

def main():
    #model = tfhub.load('https://bit.ly/metrabs_l')
    model = tf.saved_model.load('/home/lyn/HitLyn/metrabs/models/metrabs_rn101_y4')
    skeleton = 'smpl_24'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    frame_batches = tf.data.Dataset.from_generator(frames_from_video, tf.uint8, [None, None,3]).batch(24).prefetch(1)

    for frame_batch in frame_batches:
        pred = model.detect_poses_batched(frame_batch, skeleton=skeleton, default_fov_degrees=55, max_detections=1, detector_threshold=0.2)
        for frame, boxes, poses3d in zip(frame_batch.numpy(), pred['boxes'].numpy(), pred['poses3d'].numpy()):
            write2file(np.squeeze(poses3d), OUTPUT)



def write2file(pose3d, file):
    with open(file, "a") as f:
        if len(pose3d) == 24:
            # embed();exit()
            poses_new = np.zeros_like(pose3d)
            # # poses_new2 = np.zeros_like(poses)
            poses_new[:,0] = pose3d[:,0]
            poses_new[:,1] = -pose3d[:,1]
            poses_new[:,2] = pose3d[:,2]
            np.savetxt(f, poses_new.flatten()[None, :], delimiter=',')
            print('Saving to txt...')
        elif len(pose3d) > 24:
            print('Detect more than one person, try next frame..')
        else:
            print('Failed to detect pose, try next frame...')



def frames_from_video():
    cap = cv2.VideoCapture(INPUT)
    while (frame_bgr := cap.read()[1]) is not None:
        yield frame_bgr[..., ::-1]



if __name__ == '__main__':
    main()
