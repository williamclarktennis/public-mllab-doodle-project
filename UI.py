import tkinter as tk
from tkinter import Canvas, Label, Frame
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt

class DrawingApp:
    def __init__(self, root, num_categories = None, model = None): 
        # Things needed for prediction
        self.model = model

        # Get the array of category names
        if(model is not None):
            with open(r'category_names.txt', 'r') as file:
                self.category_names = [line.strip() for line in file]
            # We only care about the first num_categories categories:
            self.category_names = self.category_names[:num_categories]

        self.root = root
        self.root.title("Doodle recognition")
        self.root.configure(bg="#2e2e2e")  # Dark gray background

        if(model is not None):
            # Text display (top)
            text_head =  "Draw something out of the following:\n" + str(self.category_names[0]) + ", " + ", ".join(self.category_names[1:])
            self.text_label_all_cat = Label(root, text = text_head, font=("Arial", 14), justify="left", fg="white",bg="#2e2e2e", wraplength=600)
            self.text_label_all_cat.pack(side=tk.TOP)

        # Drawing frame (to add a border around the canvas)
        self.canvas_frame = Frame(root, bg="blue", padx=5, pady=5)
        self.canvas_frame.pack(side=tk.LEFT, padx=20, pady=20)

        # Canvas for drawing (left side) inside the frame
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = Canvas(root, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(side=tk.LEFT)

        # Add a Clear button below the canvas
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, pady=1)
        
        # Text display (middle)
        self.text_label_prob = Label(root, text="Probabilities", font=("Arial", 14), justify="left", fg="white",bg="#2e2e2e")
        self.text_label_prob.pack(side=tk.RIGHT, padx=20, pady=20)

        # Text display (right side)
        self.text_label = Label(root, text="Draw on the left!\nRight-click to erase.", font=("Arial", 14), justify="left", fg="white",bg="#2e2e2e")  # Match root background)
        self.text_label.pack(side=tk.RIGHT, padx=20, pady=20)


        # Image for drawing (to extract pixels)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Mouse event bindings
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)  # Left-click drag
        self.canvas.bind("<B3-Motion>", self.erase_on_canvas)  # Right-click drag

        # Start update loop
        self.update_bitmap()

    def draw_on_canvas(self, event):
        r = 8  # Brush radius
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="black", outline="black")
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=0)

        # Update label text
        # self.text_label.config(text="You're drawing!")

    def erase_on_canvas(self, event):
        r = 12  # Eraser radius
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="white", outline="white")
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

        # Update label text
        # self.text_label.config(text="Erasing...")

    def clear_canvas(self):
        self.canvas.delete("all")  # Clear the visible canvas
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)  # Reset image
        self.draw = ImageDraw.Draw(self.image)  # Recreate drawing context

    def update_bitmap(self):
        # Resize to 28x28 and invert colors (black = 1, white = 0)
        small_image = self.image.resize((28, 28), Image.LANCZOS)
            
        inverted = ImageOps.invert(small_image)

        bitmap = np.array(inverted)/255
        assert bitmap.shape == (28,28)

        if(self.model is not None):
            prediction = self.model(bitmap.reshape((1,28,28,1)))        # The NN is picky and needs this shape if you use __call__ (faster than .predict)

            # The output has shape (1, num_categories)
            self.update_text(np.array(prediction).flatten())


        # Repeat every __ ms
        self.root.after(100, self.update_bitmap)

    def update_text(self, predictions, len_leaderboard = 6):
        len_leaderboard = min(len(self.category_names), len_leaderboard)       # There should be more categories than categories to be displayed in the drawing window!

        # Pick the len_leaderboard categories which are the most likely, sorted from most likely to less likely 
        highest_pred = np.argsort(predictions)[-1:-len_leaderboard-1:-1]
        
        # Round the probabilities and sale them to [0,100]
        predictions = np.round(predictions*100, 2)

        # Prepare the text
        text_to_display = self.category_names[highest_pred[0]]
        prob_to_display = str(predictions[highest_pred[0]]) + "%"
        for i in range(1, len_leaderboard):
            text_to_display += "\n" + self.category_names[highest_pred[i]]
            prob_to_display += "\n" + str(predictions[highest_pred[i]]) + "%"

        self.text_label.config(text = text_to_display)
        self.text_label_prob.config(text = prob_to_display)




if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
