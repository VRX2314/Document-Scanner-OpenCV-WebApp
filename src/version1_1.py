import cv2
import streamlit as st
import numpy as np
import time


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


# def capture():
#     global capstate
#     if capstate == 0:
#         capstate = 1

#     elif capstate == 1:
#         capstate = 2

#     else:
#         pass
#     print(capstate)


cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 720
widthImg = 1280

firstBoot = True

################################################## Streamlit UI ##################################################

# st.set_page_config(layout="wide")
st.title("Document Scanner v1.1")
st.text("Github: VRX2314")
# st.text("Roll No: C093")

left_column, right_column = st.columns([1, 2.5])

with left_column:
    frame_vanilla = st.empty()

    roi_label = "Toggle Region of Intrest"
    roi_button = st.toggle(f"{roi_label}")

    if roi_button:
        roi_w_factor = st.slider("ROI Horizontal", 1, 10, 10)
    else:
        roi_w_factor = st.empty()
        roi_w_factor = widthImg

    left_1, left_2 = st.columns(2)

    with left_1:
        frame_thresh = st.empty()

    with left_2:
        frame_contours = st.empty()

    thresh1 = st.slider("Threshold 1", 1, 255, 200)
    thresh2 = st.slider("Threshold 2", 1, 255, 200)

    frame_big_C = st.empty()

img_white = np.full((heightImg * 4, widthImg * 2, 3), 255, np.uint8)

with right_column:
    frame_print = st.empty()
    # right_1, right_2, right_3 = st.columns(3, gap="medium")
    # with right_1:
    #     capture = st.button("📸\n\nCapture", use_container_width=True)
    # with right_2:
    #     print_b = st.button("🖨️\n\nPrint", use_container_width=True)
    #     adaptive = st.checkbox("Use Threshold")
    # with right_3:
    #     reset = st.button("❌\n\nClear", use_container_width=True)
    st.text("")
    st.text("")
    print_b = st.button("🖨️\n\nPrint", use_container_width=True)

frame_print.image(img_white)

################################################## OpenCV Logic ##################################################

while cap.isOpened():
    ret, img = cap.read()

    img = cv2.resize(img, (widthImg, heightImg))
    img[0:heightImg, : widthImg // roi_w_factor] = 255
    img[0:heightImg, widthImg - widthImg // roi_w_factor :] = 255
    normalizedImg = np.zeros((heightImg, widthImg, 3))
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    frame_vanilla.image(img, channels="BGR", caption="Input Stream")

    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgBuffer = np.zeros((heightImg, widthImg, 3), np.uint8)
    emptyBuffer = np.zeros((heightImg, widthImg, 3), np.uint8)

    # CONVERT IMAGE TO GRAY SCALE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ADD GAUSSIAN BLUR
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # APPLY CANNY BLUR
    imgThreshold = cv2.Canny(imgBlur, thresh1, thresh2)
    kernel = np.ones((5, 5))

    # APPLY DILATION
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)

    # APPLY EROSION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    frame_thresh.image(imgThreshold, caption="Thresholded Edges")

    # COPY IMAGE FOR DISPLAY PURPOSES
    imgContours = img.copy()
    imgBigContour = img.copy()

    # FIND ALL CONTOURS
    contours, hierarchy = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # DRAW ALL DETECTED CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    frame_contours.image(imgContours, channels="BGR", caption="Contours")

    # FIND THE BIGGEST CONTOUR
    biggest, maxArea = biggestContour(contours)
    # print(capture)
    # if capture:
    #     print(capstate)
    #     if capstate == 0:
    #         capstate = 1

    widthCanvas, heightCanvas, c = img_white.shape
    # IF A CONTINUOS CONTOUR IS FOUND
    if biggest.size != 0:
        biggest = reorder(biggest)

        # DRAW THE BIGGEST CONTOUR
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)

        # PREPARE POINTS FOR WARP
        imgBigContour = drawRectangle(imgBigContour, biggest, 2)

        # PREPARE POINTS FOR WARP
        pts1 = np.float32(biggest)
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[
            20 : imgWarpColored.shape[0] - 20, 20 : imgWarpColored.shape[1] - 20
        ]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # ADD SCAN IMAGE TO THE WHITE CANVAS IMAGE

        img_white[100 : heightImg + 100, 640:1920] = cv2.cvtColor(
            imgAdaptiveThre, cv2.COLOR_GRAY2BGR
        )
        img_white[heightCanvas - 820 : heightCanvas - 100, 640:1920] = imgWarpColored

    # KEEP OUTPUT AS NOT DETECTED
    else:
        imgBigContour = cv2.putText(
            imgBlank,
            "Not Detected",
            (widthImg // 2 - 200, heightImg // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        img_white[heightCanvas - 820 : heightCanvas - 100, 640:1920] = imgBlank
        img_white[100 : heightImg + 100, 640:1920] = imgBlank

    frame_big_C.image(imgBigContour, channels="BGR", caption="Card Detection")

    frame_print.image(img_white, use_column_width=True, channels="BGR")

    if print_b:
        cv2.imwrite(f"./{time.time()}.png", img_white)
        print_b = False

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
