import matplotlib.pyplot as plt
from PIL import Image
from DigitRecognizer import *

digit_recognizer = DigitRecognizer()

# (X_train, Y_train), (X_test, Y_test) = digit_recognizer.load_data()
# digit_recognizer.train(X_train, Y_train, X_test, Y_test)

digit_recognizer.load_model()

# predict
for i in range(1,10):
    img_array = DigitRecognizer.process_image(image_path = f"digits/{i}.jpg")
    # show_prediction(img_array, i, W1, b1, W2, b2)
    prediction = digit_recognizer.make_predictions(img_array)
    print("Prediction: ", prediction)
    print("Label: ", i)
    current_image = img_array.reshape((DigitRecognizer.WIDTH, DigitRecognizer.HEIGHT)) * DigitRecognizer.SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
