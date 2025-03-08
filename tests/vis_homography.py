import kornia as K
import io
import requests
import cv2
import torch
import matplotlib.pyplot as plt


def download_image(url: str, filename: str = "") -> str:
    filename = url.split("/")[-1] if len(filename) == 0 else filename
    # Download
    bytesio = io.BytesIO(requests.get(url).content)
    # Save file
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

    return filename





if __name__ == '__main__':
    url = "https://github.com/kornia/data/raw/main/bruce.png"
    download_image(url)

    img = K.io.load_image("bruce.png", K.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW
    print(img.shape)

    # the source points are the region to crop corners
    points_src = torch.tensor(
        [
            [
                [125.0, 150.0],
                [562.0, 40.0],
                [562.0, 282.0],
                [54.0, 328.0],
            ]
        ]
    )

    # the destination points are the image vertexes
    h, w = 64, 128  # destination size
    points_dst = torch.tensor(
        [
            [
                [0.0, 0.0],
                [w - 1.0, 0.0],
                [w - 1.0, h - 1.0],
                [0.0, h - 1.0],
            ]
        ]
    )

    # compute perspective transform
    M: torch.tensor = K.geometry.get_perspective_transform(points_src, points_dst)

    # warp the original image by the found transform
    img_warp: torch.tensor = K.geometry.warp_perspective(img.float(), M, dsize=(h, w))
    print(img_warp.shape)

    # convert back to numpy
    img_np = K.tensor_to_image(img)
    img_warp_np = K.tensor_to_image(img_warp)

    # draw points into original image
    for i in range(4):
        center = tuple(points_src[0, i].long().numpy())
        img_np = cv2.circle(img_np.copy(), center, 5, (0, 255, 0), -1)

    # create the plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs = axs.ravel()

    axs[0].axis("off")
    axs[0].set_title("image source")
    axs[0].imshow(img_np)

    axs[1].axis("off")
    axs[1].set_title("image destination")
    axs[1].imshow(img_warp_np)
    plt.show()



