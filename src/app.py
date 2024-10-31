# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import network
# import mnist_loader

# from matplotlib import pyplot as plt
# def getGrayScale(rgb):
#     tmp = 1-rgb
#     tmp = np.dot(tmp[...,:3], [1, 0, 0])
#     fig.add_subplot(1,2,2)
#     plt.imshow(tmp, cmap=plt.get_cmap('gray'))
#     tmp=tmp.flatten()
#     return np.reshape(tmp,(len(tmp),1))
# if __name__=="__main__":
#     training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
#     fig=plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.imshow(test_data[25][0].reshape(28,28), cmap=plt.get_cmap('gray'))
#     net=network.Network(saves=("data/save/weights.npy","data/save/biases.npy"))
#     img = mpimg.imread('data/mysample/nine.png')     
#     gray = getGrayScale(img)   
#     fig.add_subplot(1,2,2) 
#     plt.imshow(gray.reshape(28,28), cmap=plt.get_cmap('gray'))

    
#     ans=net.feedforward(gray)
#     plt.show()
#     print(np.argmax(ans))


from Tkinter import *
import numpy as np
from PIL import ImageGrab,Image
from network import Network

window = Tk()
window.title("Handwritten digit recognition")
l1 = Label()


def MyProject():
    global l1

    widget = cv
    # Setting co-ordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Image is captured from canvas and is resized to (28 X 28) px
    img = ImageGrab.grab().crop((x*2+6, y*2+6, x1*2-6, y1*2-6)).resize((28,28),Image.LANCZOS)
    img.save("data/mysample/resized.png")
    # Converting rgb to grayscale image
    img = img.convert('L')

    # Extracting pixel matrix of image and converting it to a vector of (1, 784)
    x = np.asarray(img)
    np.save("data/mysample/img.npy",x)
    vec = np.zeros((1, 784))
    k = 0
    for i in range(28):
        for j in range(28):
            vec[0][k] = x[i][j]
            k += 1

    net=Network(saves=("data/save/weights.npy","data/save/biases.npy"))

    # Calling function for prediction
    model_input=vec[0].reshape(784,1) / 255
    np.save("data/mysample/model_input.npy",model_input)
    pred = net.feedforward(model_input)

    # Displaying the result
    l1 = Label(window, text="Digit = " + str(np.argmax(pred)), font=('Algerian', 20))
    l1.place(x=230, y=420)


lastx, lasty = None, None


# Clears the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()


# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# Label
L1 = Label(window, text="Handwritten Digit Recoginition", font=('Algerian', 25), fg="blue")
L1.place(x=35, y=10)

# Button to clear canvas
b1 = Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

# Button to predict digit drawn on canvas
b2 = Button(window, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=370)

# Setting properties of canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
