import cv2
import os


def make_video_from_images(image_dir, output_video, fps=30):
    """
    Create a video from a directory of images.

    Parameters:
    image_dir (str): Path to the directory containing the images.
    output_video (str): Path to save the output video file.
    fps (int): Frames per second for the video.
    """
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png") or img.endswith(".jpg")])

    if not images:
        print("No images found in the directory!")
        return

    # Read the first image to get the width and height
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through images and write each one as a frame in the video
    for image in images:
        img_path = os.path.join(image_dir, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # Write the frame to the video

    video.release()  # Release the video writer object
    print(f"Video saved as {output_video}")


if __name__ == '__main__':
    # Wildcat image dir
    image_directory = '/home/edison/Research/omnisafe_zjy/examples/evaluations/riverine/video_inference/eval_figs_wildcat'
    output_video_path = '/home/edison/Research/omnisafe_zjy/examples/evaluations/riverine/video_inference/models_eval_video_wildcat.mp4'

    # Wabash image dir
    # image_directory = '/home/edison/Research/omnisafe_zjy/examples/evaluations/riverine/video_inference/eval_figs_wabash'
    # output_video_path = '/home/edison/Research/omnisafe_zjy/examples/evaluations/riverine/video_inference/models_eval_video_wabash.mp4'

    print(f'Starting generating video ...')

    make_video_from_images(image_directory, output_video_path, fps=10)

    print(f'Models prediction video has been saved to {output_video_path}.')
