"""Capture a screenshot of the control panel for the code documentation.

Launches the GUI, selects a representative program, lets it render, grabs the
window with PIL.ImageGrab, saves PNG, then closes. Run on a machine with a
display. Output: gui_screenshot.png (and a second view gui_screenshot_census.png).
"""
import os
import sys
import time
import tkinter as tk
from PIL import ImageGrab

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runall_gui as G


def grab(root, path):
    root.update_idletasks()
    root.update()
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    img.save(path)
    print("saved", path, img.size)


def main():
    root = tk.Tk()
    app = G.App(root)
    root.geometry("1280x900")
    root.update()
    time.sleep(0.5)
    root.update()
    # view 1: master verifier selected
    app.select("master")
    root.update()
    time.sleep(0.3)
    grab(root, os.path.join(HERE, "gui_screenshot.png"))
    # view 2: attribution control selected (shows a different figure + text)
    app.select("attrib")
    root.update()
    time.sleep(0.3)
    grab(root, os.path.join(HERE, "gui_screenshot_control.png"))
    # view 3: census selected (audit 14 general 2: the census view was missing)
    app.select("census")
    root.update()
    time.sleep(0.3)
    grab(root, os.path.join(HERE, "gui_screenshot_census.png"))
    root.destroy()


if __name__ == "__main__":
    main()
