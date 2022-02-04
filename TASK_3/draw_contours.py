import cv2

def getContours(img, imgContour):
    contours, heirachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    black = (0,0,0)
    for c in contours:
        area = cv2.contourArea(c)
        if 900 < area < 6000:
            cv2.drawContours(imgContour, c, -1, black, 2)
            p = cv2.arcLength(c, True)
            appr = cv2.approxPolyDP(c, 0.02 * p, True)
            x, y, w, h = cv2.boundingRect(appr)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h+2), (0, 255, 255), 2)
            cv2.putText(imgContour, f" A:{int(area)}", (x, y-2), cv2.FONT_HERSHEY_COMPLEX, 0.4, black, 1)
            print(f"Area = {int(area)}")