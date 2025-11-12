
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Kelias iki STOP ženklo (class_14)
image_path = "examples_gtsrb/class_14.ppm"

# Nuskaitome ir parodome
img = mpimg.imread(image_path)

plt.imshow(img)
plt.title("STOP ženklas (class_14)")
plt.axis("off")
plt.show()