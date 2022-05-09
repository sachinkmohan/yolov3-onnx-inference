import time
import cv2
import onnxruntime
import numpy as np

from PIL import Image

font2 = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 1

session = onnxruntime.InferenceSession("yolov3-10.onnx", providers=['CUDAExecutionProvider'])
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]



def frame_process(frame, input_shape=(416, 416)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image

def get_prediction(image_data, image_size):
    input = {
        inname[0]: image_data,
        inname[1]: image_size
    }
    t0 = time.time()
    boxes, scores, indices = session.run(outname, input)
    predict_time = time.time() - t0
    print("Predict Time: %ss" % (predict_time))
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
    return out_boxes, out_scores, out_classes, predict_time

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def letterbox_image2(image, size):
    iw, ih = image.shape[1], image.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, nw, nh, interpolation=Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

label =["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

#cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
#cap = cv2.VideoCapture('road.mp4')
cap = cv2.VideoCapture('untitled2.mp4')
sum_time = 0
sum_frame = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #image_data = frame_process(frame, input_shape=(416, 416))
        resized = cv2.resize(frame,(416,416))
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_data = preprocess(pil_img)
        image_size = np.array([416, 416], dtype=np.float32).reshape(1, 2)
        out_boxes, out_scores, out_classes, predict_time = get_prediction(image_data, image_size)
        sum_time += predict_time
        sum_frame += 1
        out_boxes = np.array(out_boxes).tolist()
        out_scores = np.array(out_scores).tolist()
        out_classes = np.array(out_classes).tolist()

        for i, c in reversed(list(enumerate(out_classes))):
            a = list(enumerate(out_classes))
            print(c)
            predicted_class = label[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = int(bottom)
            right = int(right)

            cv2.rectangle(resized, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
            cv2.putText(resized, label, (left, top), font2, fontScale, color=(255, 255, 0), thickness=2)
        cv2.imshow('im', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        #for i, c in reversed(list(enumerate(out_classes))):
        #    print("box:", out_boxes[i])
        #    print("score:", out_scores[i],",", label[c])
        #print("\n")

    else:
        print("-------------------------------------------------")
        print("Average Predict Time: %ss" % (sum_time / sum_frame))
        print("-------------------------------------------------\n")
        break

cap.release()
cv2.destroyAllWindows()
