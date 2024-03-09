import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model

# Load pre-trained model
model_path = r'Model23.h5'
model = load_model(model_path)

# Define classes
pl_chomeo = {
    0: 'Con mèo!',
    1: 'Con chó!',
}
default_image_path = "D:\\BTL_CACMON\\MIT_APP\\avata.jpg"
# Create Tkinter window
root = tk.Tk()
root.geometry('800x600')
root.title('Nhận diện cho mèo ')

# THÔNG TIN CÁ NHÂN
info_labels = [
    "Họ và Tên : Hoàng Mạnh Tiến",
    "Lớp 56KMT",
    "MASV : K205480106025"
]

for i, info in enumerate(info_labels):
    label = tk.Label(root, text=info ,anchor="w")
    label.pack(anchor ="w")

# Widgets
label = Label(root, background='#ffc0cb', font=('calibri', 15, 'bold'))
sign_image = Label(root)

# Function to classify image
def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((128, 128))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image / 255
    pred = model.predict([image])[0]
    pred_class = numpy.argmax(pred)
    sign = pl_chomeo[pred_class]
    label.configure(foreground='#000000', text=sign)
    messagebox.showinfo("Kết quả là", sign)


# HAM HIEN NUT PHAN LOAI
def show_classify_button(file_path):
    classify_b = Button(root, text="Phân loại",
                        command=lambda: classify(file_path),
                        padx=10, pady=10, font=('calibri', 12, 'bold'))
    classify_b.place(relx=0.5, rely=0.85, anchor=CENTER)


# UPLOAD FILE ẢNH
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:  # Nếu không có ảnh được chọn
            file_path = default_image_path  # Sử dụng ảnh mặc định
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width()/2.25), (root.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(e)
upload_image()
# BTN CHON ANH
upload = Button(root, text="Chọn ảnh !", command=upload_image, padx=10, pady=5,
                font=('calibri', 12, 'bold'), background='#CCFF00', foreground='black')
upload.place(relx=0.5, rely=0.75, anchor=CENTER)

# Display image, result label, and heading
sign_image.place(relx=0.5, rely=0.45, anchor=CENTER)
label.place(relx=0.5, rely=0.70, anchor=CENTER)
heading = Label(root, text=" Chó hay Mèo??", pady=20, font=('calibri', 17, 'bold'))
heading.configure(background='#ffc0cb', foreground='black')
heading.place(relx=0.5, rely=0.15, anchor=CENTER)

# Run Tkinter event loop
root.mainloop()
