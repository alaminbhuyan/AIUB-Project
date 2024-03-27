import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

path = r"images\CHNCXR_0001_0.jpg"

real_img = cv2.imread(path)

resize_img = cv2.resize(real_img, (128, 128))

img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)

img = img / 255
img_array = np.expand_dims(img, axis=0)

# print(img.shape)
# print(img_array.shape)

model = load_model(r"model.h5")

# print(model)
pred = model.predict(img_array)
# pred = model.predict_generator(img_array)

print(pred.shape)

preds_test_thresh = (pred >= 0.7).astype(np.uint8)


print(preds_test_thresh.shape)
print(preds_test_thresh.min())
print(preds_test_thresh.max())

mask = preds_test_thresh[0, :, :, 0]

plt.imshow(resize_img, cmap='gray')
plt.imshow(mask, cmap='Reds', alpha=0.3)
plt.show()

# cv2.imshow("img", img)
# cv2.waitKey()
# cv2.destroyAllWindows()