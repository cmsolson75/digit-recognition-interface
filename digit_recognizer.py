from tkinter import *
from PIL import Image
import io
import torch
import numpy as np
from architecture import LeNet, pre_processing


class DigitRecognizer:
    """
    Some UI setup and event handling patterns were initially adapted from:
    https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06
    """
    DEFAULT_PEN_SIZE = 30.0
    DEFAULT_COLOR = "black"

    def __init__(self):
        """Initialize the GUI and model."""
        self.device = "cpu"
        self.model = LeNet().to(self.device)
        self.model.load_state_dict(
            torch.load("model/leNet_model.pth", map_location=torch.device(self.device))
        )

        self.root = Tk()
        self.root.title("Digit Recognition")

        self.clear_button = Button(self.root, text="CLEAR", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=1)

        self.predict_button = Button(
            self.root, text="PREDICT", command=self.process_and_predict
        )
        self.predict_button.grid(row=0, column=2)

        self.result_label = Label(self.root, text="Draw a digits", font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=3)

        self.canvas = Canvas(self.root, bg="white", width=280, height=280)
        self.canvas.grid(row=1, columnspan=3)

        self.setup()
        self.root.mainloop()

    def setup(self):
        """
        Setup event bindings for painting.
        """
        self.old_x = None
        self.old_y = None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def clear_canvas(self):
        """
        Clear Canvas of all paint
        """
        self.canvas.delete("all")

    def process_and_predict(self):
        """
        Process canvas drawing and make a prediction.
        """
        self.canvas.update()
        ps = self.canvas.postscript(colormode="color")
        img = Image.open(io.BytesIO(ps.encode("utf-8"))).convert("L")
        img_array = np.asarray(img)
        img_array = 255 - img_array

        img_array = img_array.astype(np.float32) / 255
        prediction = self.predict(img_array)

        self.result_label.config(text=prediction)

    def predict(self, img: np.ndarray) -> str:
        """
        Use the trained model to predict the digit.
        """
        classes = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        self.model.eval()

        img = pre_processing(img.copy()).type(torch.float32).unsqueeze(0)
        with torch.no_grad():
            img = img.to(self.device)
            pred = self.model(img)
            prediction_label = classes[pred[0].argmax(0)]
        
        return prediction_label

    def paint(self, event):
        """
        Capture user drawing on canvas.
        """
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=self.DEFAULT_PEN_SIZE,
                fill=self.DEFAULT_COLOR,
                capstyle=ROUND,
                smooth=True,
                splinesteps=36,
            )

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        """
        Reset drawing coordinates when user stops drawing.
        """
        self.old_x, self.old_y = None, None


if __name__ == "__main__":
    DigitRecognizer()
