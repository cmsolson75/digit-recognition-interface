from tkinter import *
from PIL import Image
import io
import numpy as np

import matplotlib.pyplot as plt

class Paint:
    # Code Adapted from https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

    DEFAULT_PEN_SIZE = 20.0
    DEFAULT_COLOR = 'black'
    TEXT = "HI"

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)

        self.predict_button = Button(self.root, text='predict', command=self.get_canvas_array)
        self.predict_button.grid(row=0, column=2)

        self.my_label = Label(self.root, text="press the button")
        self.my_label.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=560, height=560)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        # Have alternate start command
        self.root.mainloop()
    
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def update_text(self, text: str):
        self.my_label.config(text=text)


    def use_pen(self):
        self.activate_button(self.pen_button)
    
    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
    
    def activate_button(self, button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        button.config(relief=SUNKEN)
        self.active_button = button
        self.eraser_on = eraser_mode
    
    def get_canvas_array(self):
        self.c.update()
        ps = self.c.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8'))).convert("L")
        img_array = np.asarray(img)
        self.update_text(str(img_array.shape))

        print(img_array.shape)

    
    def paint(self, event):
        # self.line_width = 
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=True, splinesteps=36)

        self.old_x = event.x
        self.old_y = event.y
    def reset(self, event):
        self.old_x, self.old_y = None, None
    

if __name__ == "__main__":
    Paint()

