import cv2
import time
import numpy
cap = cv2.VideoCapture(1)
cap.set(3, 480)  # set width of the frame
cap.set(4, 640)  # set height of the frame

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
age_list = ['0-6', '7-12', '13-18', '19-24', '25-34', '35-44', '45-59', '60-100']
gender_list = ['M', 'F']
count_age_list = [0] * 8
count_gender_list = [0] * 2
temp = [0] * 2


def settings_for_ads(win_name):
    cv2.waitKey(1000)
    cv2.destroyWindow(win_name)
    for element in range(len(count_age_list)):
        count_age_list[element] = 0
    for element in range(len(count_gender_list)):
        count_gender_list[element] = 0
    start1 = time.time()
    return start1


def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe(r'deploy_age.prototxt',
                                       r'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe(r'deploy_gender.prototxt',
                                          r'gender_net.caffemodel')
    return (age_net, gender_net)


def video_detector(age_net, gender_net):
    global count_age_list, count_gender_list, start
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()
    time_delay = 2  # in seconds
    count = 0
    count_delay = 20
    while True:
        ret, image = cap.read()
        #gray = cv2.flip(image, -1)
        face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            #minSize=(24, 24)
        )
        if len(faces) > 0:
            print("Found {} faces".format(str(len(faces))))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (128, 128, 128), 1)
            # Get Face
            face_img = image[y:y + h, h:h + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)
            temp[0] = gender_preds[0].argmax()
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)
            temp[1] = age_preds[0].argmax()
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', image)

        # showing ADs
        if len(faces) > 0:
            count += 1
            count_age_list[temp[1]] += 1
            count_gender_list[temp[0]] += 1
        if count == count_delay:
            count = 0
            k_age = numpy.argmax(count_age_list)
            k_gender = gender_list[numpy.argmax(count_gender_list)]
            if k_age == 0:
                cv2.imshow('0-6' + k_gender, cv2.imread('Mpics/0-6.jpg'))
                start = settings_for_ads(win_name='0-6' + k_gender)
            elif k_age == 1:
                cv2.imshow('7-12' + k_gender, cv2.imread('Mpics/7-12.jpg'))
                start = settings_for_ads(win_name='7-12' + k_gender)
            elif k_age == 2:
                cv2.imshow('13-18' + k_gender, cv2.imread('Mpics/13-18.jpg'))
                start = settings_for_ads(win_name='13-18' + k_gender)
            elif k_age == 3:
                cv2.imshow('19-24' + k_gender, cv2.imread('Mpics/19-24.jpg'))
                start = settings_for_ads(win_name='19-24' + k_gender)
            elif k_age == 4:
                cv2.imshow('25-34' + k_gender, cv2.imread('Mpics/25-34.jpg'))
                start = settings_for_ads(win_name='25-34' + k_gender)
            elif k_age == 5:
                cv2.imshow('35-44' + k_gender, cv2.imread('Mpics/35-44.jpg'))
                start = settings_for_ads(win_name='35-44' + k_gender)
            elif k_age == 6:
                cv2.imshow('45-59' + k_gender, cv2.imread('Mpics/45-59.jpg'))
                start = settings_for_ads(win_name='45-59' + k_gender)
            elif k_age == 7:
                cv2.imshow('60-100' + k_gender, cv2.imread('Mpics/60-100.jpg'))
                start = settings_for_ads(win_name='60-100' + k_gender)
        '''
        if time.time() - start <= time_delay and len(faces) > 0:
            count_age_list[temp[1]] += 1
            count_gender_list[temp[0]] += 1
        elif time.time() - start > time_delay and len(faces) > 0:
            k_age = numpy.argmax(count_age_list)
            if k_age == 0:
                cv2.imshow('0-6', cv2.imread('Mpics/0-6.jpg'))
                start = settings_for_ads(win_name='0-6')
            elif k_age == 1:
                cv2.imshow('7-12', cv2.imread('Mpics/7-12.jpg'))
                start = settings_for_ads(win_name='7-12')
            elif k_age == 2:
                cv2.imshow('13-18', cv2.imread('Mpics/13-18.jpg'))
                start = settings_for_ads(win_name='13-18')
            elif k_age == 3:
                cv2.imshow('19-24', cv2.imread('Mpics/19-24.jpg'))
                start = settings_for_ads(win_name='19-24')
            elif k_age == 4:
                cv2.imshow('25-34', cv2.imread('Mpics/25-34.jpg'))
                start = settings_for_ads(win_name='25-34')
            elif k_age == 5:
                cv2.imshow('35-44', cv2.imread('Mpics/35-44.jpg'))
                start = settings_for_ads(win_name='35-44')
            elif k_age == 6:
                cv2.imshow('45-59', cv2.imread('Mpics/45-59.jpg'))
                start = settings_for_ads(win_name='45-59')
            elif k_age == 7:
                cv2.imshow('60-100', cv2.imread('Mpics/60-100.jpg'))
                start = settings_for_ads(win_name='60-100')
        '''
        # 0xFF is a hexadecimal constant which is 11111111 in binary.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)
    cap.release()
    cv2.destroyAllWindows()