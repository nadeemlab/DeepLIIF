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

from Dataset_Generation.ReadBioformatImage import read_bf_image, write_bf_image

from matplotlib.image import imsave


class RegistrationApp:
    def __init__(self, window, window_title):

        self.hematoxylin = [0.651, 0.701, 0.29]
        self.eosin = [0.216, 0.801, 0.558]
        self.color_deconv = np.array([[0.65, 0.70, 0.29], [0.216, 0.801, 0.558], [0.316, 0.598, 0.737]])

        self.cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['green', 'white'])
        self.cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['maroon', 'white'])
        self.cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])

        self.maxIntensity = 255
        self.phi = 1
        self.theta = 1

        self.alignment_values = [0, 0]
        self.start_index = [0, 0]
        self.zoom_value = 0

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
        # self.canvas_base.grid(row=0, column=0)
        self.canvas_base.place(x=self.padding, y=self.padding)

        self.canvas_moving = tk.Canvas(self.window, width=self.moving_canvas_size, height=self.moving_canvas_size)
        # self.canvas_moving.grid(row=0, column=1)
        self.canvas_moving.place(x=self.base_canvas_size+self.padding, y=self.padding)

        self.canvas_blended = tk.Canvas(self.window, width=self.blended_canvas_size, height=self.blended_canvas_size)
        # self.canvas_blended.grid(row=1, column=0)
        self.canvas_blended.place(x=self.padding, y=self.moving_canvas_size+self.padding)

        self.initial_load()

        self.load_base_image_button = tk.Button(self.window, text="Open Base Image", command=lambda event='base': self.open_images(event), width=15, height=2)
        # self.next_button.grid(row=1, column=1)
        self.window.bind('l', self.open_images)
        self.load_base_image_button.place(y=int(screen_height / 8), x=int(screen_height * 3 / 4) - 115)

        self.load_moving_image_button = tk.Button(self.window, text="Open Moving Image", command=lambda event='moving': self.open_images(event), width=15, height=2)
        # self.next_button.grid(row=1, column=1)
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
        # self.save_button.grid(row=1, column=2)
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
        # self.down_button.grid(row=2, column=1)
        self.window.bind('<Down>', self.down)
        self.down_button.place(y=int(screen_height / 8) + 280, x=int(screen_height * 3 / 4) - 10)
        # self.down_button.pack()

        self.up_button = tk.Button(self.window, text="Up", command=self.up, width=10, height=2)
        # self.up_button.grid(row=2, column=2)
        self.window.bind('<Up>', self.up)
        self.up_button.place(y=int(screen_height / 8) + 200, x=int(screen_height * 3 / 4) - 10)
        # self.up_button.pack(anchor=tkinter.CENTER)

        self.left_button = tk.Button(self.window, text="Left", command=self.left, width=10, height=2)
        # self.left_button.grid(row=3, column=1)
        self.window.bind('<Left>', self.left)
        self.left_button.place(y=int(screen_height / 8) + 240, x=int(screen_height * 3 / 4) - 85)

        self.right_button = tk.Button(self.window, text="Right", command=self.right, width=10, height=2)
        # self.right_button.grid(row=3, column=2)
        self.window.bind('<Right>', self.right)
        self.right_button.place(y=int(screen_height / 8) + 240, x=int(screen_height * 3 / 4) + 65)

        self.zoom_in_button = tk.Button(self.window, text="Zoom in", command=self.zoom_in, width=10, height=2)
        # self.right_button.grid(row=3, column=2)
        self.window.bind('z', self.zoom_in)
        self.zoom_in_button.place(y=int(screen_height / 8) + 340, x=int(screen_height * 3 / 4) - 65)

        self.zoom_out_button = tk.Button(self.window, text="Zoom out", command=self.zoom_out, width=10, height=2)
        # self.right_button.grid(row=3, column=2)
        self.window.bind('z', self.zoom_in)
        self.zoom_out_button.place(y=int(screen_height / 8) + 340, x=int(screen_height * 3 / 4) + 55)

        zoom_x_var = tk.StringVar()
        self.label_zoom_x = tk.Label(window, textvariable=zoom_x_var)
        zoom_x_var.set('Zoom X Value:')
        self.label_zoom_x.place(y=int(screen_height / 8) + 400, x=int(screen_height * 3 / 4) - 65)

        self.zoom_x_text_box = tk.Entry(window, width=2)
        self.zoom_x_text_box.place(y=int(screen_height / 8) + 400, x=int(screen_height * 3 / 4) + 55)
        self.zoom_x_text_box.insert(0, '3')

        zoom_y_var = tk.StringVar()
        self.label_zoom_y = tk.Label(window, textvariable=zoom_y_var)
        zoom_y_var.set('Zoom Y Value:')
        self.label_zoom_y.place(y=int(screen_height / 8) + 420, x=int(screen_height * 3 / 4) - 65)

        self.zoom_y_text_box = tk.Entry(window, width=2)
        self.zoom_y_text_box.place(y=int(screen_height / 8) + 420, x=int(screen_height * 3 / 4) + 55)
        self.zoom_y_text_box.insert(0, '4')

        self.moving_val = 1
        self.moving_val_bulk = 10
        # self.window.bind('z', self.zoom)
        # self.window.bind('x', self.zoom_x)
        # self.window.bind('y', self.zoom_y)
        self.window.mainloop()


    def open_images(self, event):
        self.image_index = 0
        self.filename = fd.askopenfilename()
        if event == 'base':
            self.orig_base_img = Image.open(self.filename)
            self.base_img = self.orig_base_img.copy()
            self.base_img = self.base_img.resize((self.base_canvas_size, self.base_canvas_size))
            self.tk_base_img = ImageTk.PhotoImage(master=self.window, image=self.base_img)
            self.canvas_base.create_image(int(self.base_canvas_size / 2) + self.padding,
                                          int(self.base_canvas_size / 2) + self.padding, anchor=tk.CENTER,
                                          image=self.tk_base_img)
        elif event == 'moving':
            self.orig_moving_img = Image.open(self.filename)
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
        # self.blended_img = Image.new("RGBA", base_image_copy.size)
        # self.blended_img = Image.alpha_composite(self.blended_img, base_image_copy)
        # self.blended_img = Image.alpha_composite(self.blended_img, moving_image_copy)
        self.blended_img = Image.blend(base_image_copy, moving_image_copy, 0.5)
        self.tk_blended_img = ImageTk.PhotoImage(master=self.window, image=self.blended_img)
        self.canvas_blended.create_image(int(self.blended_canvas_size / 2) + self.padding,
                                         int(self.blended_canvas_size / 2) + self.padding, anchor=tk.CENTER,
                                         image=self.tk_blended_img)

    def reload_moving_image(self):
        crop_aligned = Image.new('RGB', self.moving_img.size)
        aligned_image = self.moving_img.copy()
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
            zoomed_image = crop_aligned.crop((zoom_y_val * self.zoom_value,
                                              zoom_x_val * self.zoom_value,
                                              image_size[1] - zoom_y_val * self.zoom_value,
                                              image_size[0] - zoom_x_val * self.zoom_value)).resize((image_size[1], image_size[0]))
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
        # self.crop_moving.show()
        self.create_blended_image()

    def blend_image_files(self, filename1, filename2):
        background = Image.open(filename2)
        overlay = Image.open(filename1)

        background = background.convert("RGBA")
        overlay = overlay.convert("RGBA")

        new_img = Image.blend(background, overlay, 0.4)
        return new_img

    def blend_images(self, background, overlay):
        # background = background.convert("RGBA")
        # overlay = overlay.convert("RGBA")
        # overlay = cv2.cvtColor(1-cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        #
        # new_img = cv2.addWeighted(background, 0.3, overlay, 0.7, 0)
        size_x = min(background.shape[0], overlay.shape[0])
        size_y = min(background.shape[1], overlay.shape[1])
        background = background[:size_x, :size_y]
        overlay = overlay[:size_x, :size_y]
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        # background[:,:,2] = 0
        background[:,:,1] = 0
        # new_img = np.zeros((size_x, size_y, 3))
        # new_img[:,:,2] = overlay
        # new_img[:,:,1] = background
        new_img = cv2.addWeighted(background, 0.7, overlay, 0.3, 0)
        return new_img

    def save_channel(self, img, filename, channel):
        # Create an artificial color close to the orginal one
        # cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['darkmagenta', 'white'])
        # cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['teal', 'white'])
        # cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])

        if channel == 'hema':
            selected_cmap = self.cmap_hema
        elif channel == 'eosin':
            selected_cmap = self.cmap_eosin
        else:
            selected_cmap = self.cmap_dab

        my_dpi = 200
        fig = plt.figure(figsize=(self.width/my_dpi, self.height/my_dpi), dpi=my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, cmap=selected_cmap)
        fig.savefig(filename, dpi=my_dpi)

    def save_image(self):
        print('SAVE CLICKED')
        #tiff.imsave(self.output_dir + 'Scene' + self.scene_number + '_brown_' + str(self.start_index_i) + '_' + str(self.end_index_i) + '_' + str(
        #   self.start_index_j) + '_' + str(self.end_index_j) + '.tif', self.crop_brown)
        # imsave(self.output_dir +
        #             str(self.start_index_i) + '_' +
        #             str(self.start_index_j) + '_' +
        #             str(self.start_index_i + self.alignment_values[1]) + '_' +
        #             str(self.start_index_j + self.alignment_values[0])
        #             + '_image1.png', self.crop_brown)
        # imsave(self.output_dir +
        #             str(self.start_index_i) + '_' +
        #             str(self.start_index_j) + '_' +
        #             str(self.start_index_i + self.alignment_values[1]) + '_' +
        #             str(self.start_index_j + self.alignment_values[0])
        #             + '_image4.png', self.crop_blue)
        files = [('All Files', '*.*'),
                 ('PNG Files', '*.png'),
                 ('TIF Files', '*.tif'),
                 ('TIFF Files', '*.tiff')]
        new_filename = self.filename.split('/')[-1]
        last_dot_index = new_filename.rfind('.')
        save_file = fd.asksaveasfile(mode='w', initialfile=new_filename[0:last_dot_index] + '_registered' + new_filename[last_dot_index:],filetypes=files, defaultextension=files)
        # save_file.write(np.asarray(self.crop_moving))
        abs_path = os.path.abspath(save_file.name)
        self.crop_moving.save(abs_path)
        # self.crop_moving.save(save_file, new_filename[last_dot_index+1:])
        # imsave(self.output_dir + self.available_files[self.image_index]['name']
        #             + '_hematoxylin.tif', self.crop_blue)

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

    def zoom_y(self):
        print('in zoom')
        self.image_size = self.blue_img.shape
        self.blue_img = cv2.resize(self.blue_img[:self.image_size[0], 1:self.image_size[1] - 1], (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        self.reload_moving_image()

    def zoom_x(self):
        print('in zoom')
        self.image_size = self.blue_img.shape
        self.blue_img = cv2.resize(self.blue_img[1:self.image_size[0]-1, :self.image_size[1]], (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        self.reload_moving_image()

    def zoom_in(self):
        self.zoom_value += 1
        self.reload_moving_image()

    def zoom_out(self):
        self.zoom_value -= 1
        self.reload_moving_image()

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def next_starting(self):
        self.image_index += 1
        # self.start_index_j += self.height
        # if self.start_index_j >= self.cols:
        #     self.start_index_j = 0
        #     # self.alignment_values[1] = 0
        #     self.start_index_i += self.width
        #     if self.start_index_i >= self.rows:
        #         return
            # else:
            #     self.alignment_values[0] = 0

    def next_image(self):
        print('-----------------------------------------')
        print(self.start_index_i, self.start_index_j)
        self.next_starting()
        print(self.start_index_i, self.start_index_j)
        self.create_mask_image()


if __name__ == '__main__':
    reg_app = RegistrationApp(window=tk.Tk(), window_title="Image Alignment")