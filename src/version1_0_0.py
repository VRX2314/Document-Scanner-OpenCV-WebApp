import threading
import av
import cv2
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np

from streamlit_webrtc import webrtc_streamer


# class VideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.roi_h = 1
#         self.roi_w = 1

#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         heigh, width, c = img.shape
#         # img[0:height, width - width // self.roi_w :] = 255
#         # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#         img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

#         return img


lock = threading.Lock()
img_normalized = {"img": None}


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        (0, 255, 0),
        thickness,
    )

    return img


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def vanilla_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    height, width, c = img.shape
    img[0:height, width - width // roi_w :] = 255
    img[0:height, : width // roi_w] = 255
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    with lock:
        img_normalized["img"] = img

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def not_vanilla(frame):
    with lock:
        img = img_normalized["img"]

    return img


left_column, right_column = st.columns(2)

with right_column:
    fig_main = st.empty()

with left_column:
    ctx = webrtc_streamer(key="example", video_frame_callback=vanilla_callback)

    if ctx.video_transformer:
        roi_h = st.slider("ROI Vertical", 1, 10, 1)
        roi_w = st.slider("ROI Horizontal", 1, 10, 8)

        # cty = webrtc_streamer(
        #     key="example1",
        #     desired_playing_state=True,
        #     video_frame_callback=not_vanilla,
        # )

    fig_place = st.empty()
    fig, ax = plt.subplots(2, 2)

    thresh1 = st.slider("Threshold 1", 1, 250, 200)
    thresh2 = st.slider("Threshold 2", 1, 250, 200)

    while ctx.state.playing:
        with lock:
            img = img_normalized["img"]
        if img is None:
            continue
        height, width, c = img.shape

        imgBlank = np.zeros((height, width, 3), np.uint8)

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgThreshold = cv2.Canny(imgBlur, thresh1, thresh2)
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

        imgContours = img.copy()
        imgBigContour = img.copy()

        contours, hierarchy = cv2.findContours(
            imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        biggest, maxArea = biggestContour(contours)

        if biggest.size != 0:
            biggest = reorder(biggest)
            cv2.drawContours(
                imgBigContour, biggest, -1, (0, 255, 0), 20
            )  # DRAW THE BIGGEST CONTOUR
            imgBigContour = drawRectangle(imgBigContour, biggest, 2)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32(
                [[0, 0], [width, 0], [0, height], [width, height]]
            )  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

            # REMOVE 20 PIXELS FORM EACH SIDE
            imgWarpColored = imgWarpColored[
                20 : imgWarpColored.shape[0] - 20, 20 : imgWarpColored.shape[1] - 20
            ]
            imgWarpColored = cv2.resize(imgWarpColored, (width, height))

            # APPLY ADAPTIVE THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        else:
            imgWarpColored = imgBlank
            imgAdaptiveThre = imgBlank

        ax[0][0].cla()
        ax[0][0].imshow(imgThreshold, cmap="gray")
        ax[0][0].axis("off")

        ax[0][1].cla()
        ax[0][1].imshow(cv2.cvtColor(imgContours, cv2.COLOR_BGR2RGB))
        ax[0][1].axis("off")
        fig_place.pyplot(fig)

        ax[1][0].cla()
        ax[1][0].imshow(cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2RGB))
        ax[1][0].axis("off")

        ax[1][1].cla()
        ax[1][1].imshow(cv2.cvtColor(imgAdaptiveThre, cv2.COLOR_BGR2RGB))
        ax[1][1].axis("off")
        fig_place.pyplot(fig)
