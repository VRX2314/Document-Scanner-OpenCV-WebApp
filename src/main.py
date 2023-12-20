# Roll No: C093
# Name: Vansh Shah
# Topic: Realtime Document Scanner

import cv2
import streamlit as st
import numpy as np
import time
import pytesseract
from PIL import Image


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


def drawRectangle(img, biggest, thickness, colstate):
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        colstate,
        thickness,
    )
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        colstate,
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        colstate,
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        colstate,
        thickness,
    )

    return img


################################################## Streamlit UI ##################################################

st.title("Document Scanner v1.1")
st.text("Github: VRX2314")
src = st.selectbox("Select Source", [0, 1, "Upload"])

if src == 0:
    cap = cv2.VideoCapture(0)
elif src == 1:
    cap = cv2.VideoCapture(1)

elif src == "Upload":
    up = st.file_uploader("Upload Image", ["png", "jpg"])
    if up != None:
        img = Image.open(up)
        img = np.array(img)

heightImg = 720
widthImg = 1280

firstBoot = True

left_column, right_column = st.columns([1, 2.5])

with left_column:
    frame_vanilla = st.empty()

    roi_label = "Toggle Region of Intrest"
    roi_button = st.toggle(f"{roi_label}")
    cont_button = st.toggle("Invert Contours")

    if roi_button:
        roi_w_factor = st.slider("ROI Factor", 1, 10, 10)
        invert_col = st.toggle("Invert ROI Colour")
    else:
        roi_w_factor = st.empty()
        roi_w_factor = widthImg
        invert_col = False

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
    st.text("")
    st.text("")
    print_b = st.button("🖨️\n\nPrint", use_container_width=True)

frame_print.image(img_white)

################################################## OpenCV Logic ##################################################

while True:
    if src != "Upload":
        ret, img = cap.read()
    else:
        try:
            img = img
        except:
            st.markdown("### ⚠️ Kindly Upload Image")
            break

    if invert_col:
        col = 0
    else:
        col = 255

    img = cv2.resize(img, (widthImg, heightImg))
    img[0:heightImg, : widthImg // roi_w_factor] = col
    img[0:heightImg, widthImg - widthImg // roi_w_factor :] = col
    img[: heightImg // roi_w_factor, 0:widthImg] = col
    img[heightImg - heightImg // roi_w_factor :, 0:widthImg] = col
    normalizedImg = np.zeros((heightImg, widthImg, 3))
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgBuffer = np.zeros((heightImg, widthImg, 3), np.uint8)
    emptyBuffer = np.zeros((heightImg, widthImg, 3), np.uint8)

    # ! Grey scale part
    try:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        img = np.stack((img,) * 3, axis=-1)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_vanilla.image(
        cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR),
        channels="BGR",
        caption="Input Stream",
    )

    # * Pre processing

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # ! Contour needs Threshold
    imgThreshold = cv2.Canny(imgBlur, thresh1, thresh2)
    kernel = np.ones((5, 5))

    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)

    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    frame_thresh.image(imgThreshold, caption="Thresholded Edges")

    imgContours = img.copy()
    imgBigContour = img.copy()

    # * Contours section (img, retriever, approx (simple for removing redundants))
    contours, hierarchy = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if cont_button:
        cols = (0, 0, 0)
    else:
        cols = (255, 255, 255)
    cv2.drawContours(imgContours, contours, -1, cols, 25)
    imgContoursGray = cv2.cvtColor(imgContours, cv2.COLOR_BGR2GRAY)

    frame_contours.image(
        cv2.cvtColor(imgContoursGray, cv2.COLOR_GRAY2BGR),
        channels="BGR",
        caption="Contours",
    )

    biggest, maxArea = biggestContour(contours)

    widthCanvas, heightCanvas, c = img_white.shape

    if biggest.size != 0:
        biggest = reorder(biggest)

        cv2.drawContours(imgBigContour, biggest, -1, (255, 255, 255), 20)

        # * Warp Logic
        imgBigContour = drawRectangle(imgBigContour, biggest, 20, cols)
        imgBigContourGray = cv2.cvtColor(imgBigContour, cv2.COLOR_BGR2GRAY)

        pts1 = np.float32(biggest)
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
        )

        # * Take contour points to warp
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgWarpColored = imgWarpColored[
            20 : imgWarpColored.shape[0] - 20, 20 : imgWarpColored.shape[1] - 20
        ]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # * Output Generation

        img_white[100 : heightImg + 100, 640:1920] = cv2.cvtColor(
            imgAdaptiveThre, cv2.COLOR_GRAY2BGR
        )
        img_white[heightCanvas - 820 : heightCanvas - 100, 640:1920] = cv2.cvtColor(
            imgWarpGray, cv2.COLOR_GRAY2BGR
        )

    # ! Default state
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
        imgWarpColored = imgBigContour
        imgBigContourGray = cv2.cvtColor(imgBigContour, cv2.COLOR_BGR2GRAY)

        img_white[heightCanvas - 820 : heightCanvas - 100, 640:1920] = imgBlank
        img_white[100 : heightImg + 100, 640:1920] = imgBlank

    frame_big_C.image(
        cv2.cvtColor(imgBigContourGray, cv2.COLOR_GRAY2BGR),
        channels="BGR",
        caption="Card Detection",
    )
    frame_print.image(img_white, use_column_width=True, channels="BGR")

    ############################################ OCR ############################################

    if print_b:
        cv2.imwrite(f"./{time.time()}.png", img_white)
        ocr_img = imgWarpColored
        text = pytesseract.image_to_data(ocr_img, output_type=pytesseract.Output.DICT)
        print(text)
        wordlist = []
        for i in range(len(text["conf"])):
            if text["conf"][i] > 80:
                wordlist.append(text["text"][i])
        ocr_box = st.text_area(label="OCR", value=wordlist)
        print_b = False

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if src != "Upload":
    cap.release()
    cv2.destroyAllWindows()
