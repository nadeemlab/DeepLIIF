import cv2
import tkinter as tk
from tkinter import filedialog as fd

import math

import imageio
from PIL import ImageTk, Image
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tifffile as tiff

from matplotlib import cm
from skimage.color import rgb2hed
import os

from matplotlib.image import imsave


class RegistrationApp:
    def __init__(self, window, window_title):

        self.alignment_values = [0, 0]
        self.start_index = [0, 0]
        self.start_index = [0, 0]
        self.zoom_value = 0
        self.rotate_value = 0

        self.padding = 10

        self.window = window
        self.window.title(window_title)

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        self.window.geometry(str(screen_height) + 'x' + str(int(screen_height * 3 / 4) + self.padding * 3))

        self.base_canvas_size = int(screen_height / 4)
        self.moving_canvas_size = int(screen_height / 4)
        self.blended_canvas_size = int(screen_height / 2)
        print(screen_height, self.base_canvas_size, self.blended_canvas_size)

        # Create a canvas that can fit the above image
        self.canvas_base = tk.Canvas(self.window, width=self.base_canvas_size, height=self.base_canvas_size)
        self.canvas_base.place(x=self.padding, y=self.padding)

        self.canvas_moving = tk.Canvas(self.window, width=self.moving_canvas_size, height=self.moving_canvas_size)
        self.canvas_moving.place(x=self.base_canvas_size+self.padding, y=self.padding)

        self.canvas_blended = tk.Canvas(self.window, width=self.blended_canvas_size, height=self.blended_canvas_size)
        self.canvas_blended.place(x=self.padding, y=self.moving_canvas_size+self.padding)

        self.initial_load()

        self.load_base_image_button = tk.Button(self.window, text="Open Base Image", command=lambda event='base': self.open_images(event), width=15, height=2)
        self.window.bind('l', self.open_images)
        self.load_base_image_button.place(y=int(screen_height / 8), x=int(screen_height * 3 / 4) - 115)

        self.load_moving_image_button = tk.Button(self.window, text="Open Moving Image", command=lambda event='moving': self.open_images(event), width=15, height=2)
        self.window.bind('l', self.open_images)
        self.load_moving_image_button.place(y=int(screen_height / 8), x=int(screen_height * 3 / 4) + 55)

        # self.load_base_wsi_button = tk.Button(self.window, text="Open Base WSI", command=self.next_image, width=15, height=2)
        # # self.next_button.grid(row=1, column=1)
        # self.window.bind('w', self.next_image)
        # self.load_base_wsi_button.place(y=int(screen_height / 8) + 40, x=int(screen_height * 3 / 4) - 115)
        #
        # self.load_moving_wsi_button = tk.Button(self.window, text="Open Moving WSI", command=self.next_image, width=15, height=2)
        # # self.next_button.grid(row=1, column=1)
        # self.window.bind('w', self.next_image)
        # self.load_moving_wsi_button.place(y=int(screen_height / 8) + 40, x=int(screen_height * 3 / 4) + 55)

        self.save_button = tk.Button(self.window, text="Save", command=self.save_image, width=10, height=2)
        self.window.bind('s', self.save_image)
        self.save_button.place(y=int(screen_height / 8) + 100, x=int(screen_height * 3 / 4) - 10)

        # self.next_button = tk.Button(self.window, text="Next", command=self.next_image, width=10, height=2)
        # # self.next_button.grid(row=1, column=1)
        # self.window.bind('n', self.next_image)
        # self.next_button.place(y=int(screen_height / 8) + 140, x=int(screen_height * 3 / 4) - 10)
        # # self.next_button.pack(anchor=tkinter.N)

        self.moving_val_text_box = tk.Entry(window, width=2)
        self.moving_val_text_box.place(y=int(screen_height / 8) + 245, x=int(screen_height * 3 / 4) + 35)
        self.moving_val_text_box.insert(0, '1')

        self.down_button = tk.Button(self.window, text="Down", command=self.down, width=10, height=2)
        self.window.bind('<Down>', self.down)
        self.down_button.place(y=int(screen_height / 8) + 280, x=int(screen_height * 3 / 4) - 10)

        self.up_button = tk.Button(self.window, text="Up", command=self.up, width=10, height=2)
        self.window.bind('<Up>', self.up)
        self.up_button.place(y=int(screen_height / 8) + 200, x=int(screen_height * 3 / 4) - 10)

        self.left_button = tk.Button(self.window, text="Left", command=self.left, width=10, height=2)
        self.window.bind('<Left>', self.left)
        self.left_button.place(y=int(screen_height / 8) + 240, x=int(screen_height * 3 / 4) - 85)

        self.right_button = tk.Button(self.window, text="Right", command=self.right, width=10, height=2)
        self.window.bind('<Right>', self.right)
        self.right_button.place(y=int(screen_height / 8) + 240, x=int(screen_height * 3 / 4) + 65)

        self.zoom_in_button = tk.Button(self.window, text="Zoom in", command=self.zoom_in, width=10, height=2)
        self.window.bind('z', self.zoom_in)
        self.zoom_in_button.place(y=int(screen_height / 8) + 340, x=int(screen_height * 3 / 4) - 65)

        self.zoom_out_button = tk.Button(self.window, text="Zoom out", command=self.zoom_out, width=10, height=2)
        self.window.bind('z', self.zoom_in)
        self.zoom_out_button.place(y=int(screen_height / 8) + 340, x=int(screen_height * 3 / 4) + 55)

        zoom_x_var = tk.StringVar()
        self.label_zoom_x = tk.Label(window, textvariable=zoom_x_var)
        zoom_x_var.set('Horizontal Value:')
        self.label_zoom_x.place(y=int(screen_height / 8) + 400, x=int(screen_height * 3 / 4) - 75)

        self.zoom_x_text_box = tk.Entry(window, width=2)
        self.zoom_x_text_box.place(y=int(screen_height / 8) + 400, x=int(screen_height * 3 / 4) + 85)
        self.zoom_x_text_box.insert(0, '3')

        zoom_y_var = tk.StringVar()
        self.label_zoom_y = tk.Label(window, textvariable=zoom_y_var)
        zoom_y_var.set('Vertical Value:')
        self.label_zoom_y.place(y=int(screen_height / 8) + 420, x=int(screen_height * 3 / 4) - 75)

        self.zoom_y_text_box = tk.Entry(window, width=2)
        self.zoom_y_text_box.place(y=int(screen_height / 8) + 420, x=int(screen_height * 3 / 4) + 85)
        self.zoom_y_text_box.insert(0, '4')

        self.rotate_right_button = tk.Button(self.window, text="Rotate Right", command=self.rotate_right, width=10, height=2)
        self.window.bind('z', self.zoom_in)
        self.rotate_right_button.place(y=int(screen_height / 8) + 480, x=int(screen_height * 3 / 4) + 70)

        self.rotate_value_text_box = tk.Entry(window, width=3)
        self.rotate_value_text_box.place(y=int(screen_height / 8) + 490, x=int(screen_height * 3 / 4) + 35)
        self.rotate_value_text_box.insert(0, '90')

        self.rotate_left_button = tk.Button(self.window, text="Rotate Left", command=self.rotate_left, width=10, height=2)
        self.window.bind('z', self.zoom_in)
        self.rotate_left_button.place(y=int(screen_height / 8) + 480, x=int(screen_height * 3 / 4) - 85)

        self.window.mainloop()

    def open_images(self, event):
        self.image_index = 0
        self.filename = fd.askopenfilename()
        if event == 'base':
            self.orig_base_img = Image.open(self.filename).convert('RGB')
            self.base_img = self.orig_base_img.copy()
            self.base_img = self.base_img.resize((self.base_canvas_size, self.base_canvas_size))
            self.tk_base_img = ImageTk.PhotoImage(master=self.window, image=self.base_img)
            self.canvas_base.create_image(int(self.base_canvas_size / 2) + self.padding,
                                          int(self.base_canvas_size / 2) + self.padding, anchor=tk.CENTER,
                                          image=self.tk_base_img)
        elif event == 'moving':
            self.orig_moving_img = Image.open(self.filename).convert('RGB')
            self.moving_img = self.orig_moving_img.copy()
            self.moving_img = self.moving_img.resize((self.moving_canvas_size, self.moving_canvas_size))
            self.crop_moving = self.moving_img.copy()
            self.tk_moving_img = ImageTk.PhotoImage(master=self.window, image=self.moving_img)
            self.canvas_moving.create_image(int(self.moving_canvas_size / 2) + self.padding,
                                          int(self.moving_canvas_size / 2) + self.padding, anchor=tk.CENTER,
                                          image=self.tk_moving_img)
        self.create_blended_image()

    def create_blended_image(self):
        base_image_copy = self.base_img.copy()
        moving_image_copy = self.crop_moving.copy()
        base_image_copy = base_image_copy.resize((self.blended_canvas_size, self.blended_canvas_size))
        moving_image_copy = moving_image_copy.resize((self.blended_canvas_size, self.blended_canvas_size))
        print(base_image_copy.size)
        print(moving_image_copy.size)
        self.blended_img = Image.blend(base_image_copy, moving_image_copy, 0.5)
        self.tk_blended_img = ImageTk.PhotoImage(master=self.window, image=self.blended_img)
        self.canvas_blended.create_image(int(self.blended_canvas_size / 2) + self.padding,
                                         int(self.blended_canvas_size / 2) + self.padding, anchor=tk.CENTER,
                                         image=self.tk_blended_img)

    def reload_moving_image(self):
        rotated_image = self.moving_img.copy()
        rotated_image = rotated_image.rotate(self.rotate_value)

        crop_aligned = Image.new('RGB', self.moving_img.size)
        aligned_image = rotated_image.copy()
        aligned_image = aligned_image.crop((max(self.start_index[1] - self.alignment_values[1], 0),
                                            max(self.start_index[0] - self.alignment_values[0], 0),
                                            min(self.moving_img.size[1] - self.alignment_values[1], self.moving_img.size[1]),
                                            min(self.moving_img.size[0] - self.alignment_values[0], self.moving_img.size[0])))

        crop_aligned.paste(aligned_image,
                           (max(self.alignment_values[1], 0), max(self.alignment_values[0], 0)))
        zoom_x_val = int(self.zoom_x_text_box.get())
        zoom_y_val = int(self.zoom_y_text_box.get())
        if self.zoom_value >= 0:
            image_size = self.moving_img.size
            zoomed_image = crop_aligned.crop((zoom_x_val * self.zoom_value,
                                              zoom_y_val * self.zoom_value,
                                              image_size[0] - zoom_x_val * self.zoom_value,
                                              image_size[1] - zoom_y_val * self.zoom_value)).resize((image_size[1], image_size[0]))
        else:
            image_size = self.moving_img.size
            zoomed_image = Image.new('RGB', image_size)
            new_cropped_image = crop_aligned.resize((image_size[1] - 2 * zoom_x_val * abs(self.zoom_value), image_size[0] - 2 * zoom_y_val * abs(self.zoom_value)))
            zoomed_image.paste(new_cropped_image,
                                (zoom_x_val * abs(self.zoom_value),
                                 zoom_y_val * abs(self.zoom_value),
                                 image_size[1] - zoom_x_val * abs(self.zoom_value),
                                 image_size[0] - zoom_y_val * abs(self.zoom_value)))

        self.crop_moving = zoomed_image
        print(self.alignment_values)
        print(self.zoom_value)
        self.create_blended_image()

    def save_image(self):
        print('SAVE CLICKED')
        files = [('All Files', '*.*'),
                 ('PNG Files', '*.png'),
                 ('TIF Files', '*.tif'),
                 ('TIFF Files', '*.tiff')]
        new_filename = self.filename.split('/')[-1]
        last_dot_index = new_filename.rfind('.')
        save_file = fd.asksaveasfile(mode='w', initialfile=new_filename[0:last_dot_index] + '_registered' + new_filename[last_dot_index:],filetypes=files, defaultextension=files)
        abs_path = os.path.abspath(save_file.name)
        self.crop_moving.save(abs_path)

    def initial_load(self):
        self.base_img = Image.fromarray(np.zeros((self.base_canvas_size, self.base_canvas_size, 3), dtype=np.uint8))
        self.tk_base_img = ImageTk.PhotoImage(master=self.window, image=self.base_img)
        self.canvas_base.create_image(int(self.base_canvas_size/2)+self.padding, int(self.base_canvas_size/2)+self.padding, anchor=tk.CENTER, image=self.tk_base_img)

        self.moving_img = Image.fromarray(np.zeros((self.moving_canvas_size, self.moving_canvas_size, 3), dtype=np.uint8))
        self.crop_moving = self.moving_img.copy()
        self.tk_moving_img = ImageTk.PhotoImage(master=self.window, image=self.moving_img)
        self.canvas_moving.create_image(int(self.moving_canvas_size/2)+self.padding, int(self.moving_canvas_size/2)+self.padding, anchor=tk.CENTER, image=self.tk_moving_img)

        self.blended_img = Image.fromarray(np.zeros((self.blended_canvas_size, self.blended_canvas_size, 3), dtype=np.uint8))
        self.tk_blended_img = ImageTk.PhotoImage(master=self.window, image=self.blended_img)
        self.canvas_blended.create_image(int(self.blended_canvas_size/2)+self.padding, int(self.blended_canvas_size/2)+self.padding, anchor=tk.CENTER, image=self.tk_blended_img)

    def down(self):
        self.moving_val = int(self.moving_val_text_box.get())
        self.alignment_values[0] += self.moving_val
        self.reload_moving_image()

    def up(self):
        self.moving_val = int(self.moving_val_text_box.get())
        self.alignment_values[0] -= self.moving_val
        self.reload_moving_image()

    def left(self):
        self.moving_val = int(self.moving_val_text_box.get())
        self.alignment_values[1] -= self.moving_val
        self.reload_moving_image()

    def right(self):
        self.moving_val = int(self.moving_val_text_box.get())
        self.alignment_values[1] += self.moving_val
        self.reload_moving_image()

    def zoom_in(self):
        self.zoom_value += 1
        self.reload_moving_image()

    def zoom_out(self):
        self.zoom_value -= 1
        self.reload_moving_image()

    def rotate_right(self):
        self.rotate_value -= int(self.rotate_value_text_box.get())
        self.reload_moving_image()

    def rotate_left(self):
        self.rotate_value += int(self.rotate_value_text_box.get())
        self.reload_moving_image()


if __name__ == '__main__':
    reg_app = RegistrationApp(window=tk.Tk(), window_title="Image Alignment")
