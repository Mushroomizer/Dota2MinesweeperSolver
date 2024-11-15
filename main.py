import os
import sys
import time

import cv2
import numpy as np
import pyautogui
import tkinter as tk
import keyboard
import threading

# Define known colors for cell classification
empty_cell_color = [50, 60, 80]
valid_green_colors = [
    [35, 42, 20],
    [30, 40, 10]
]
threshold = 20

# Debug flag
debug = False

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for both development and PyInstaller bundle """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller uses _MEIPASS to store temp files when running the exe
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Configurable threshold for color variation
def color_within_threshold(color1, color2, threshold):
    return np.all(np.abs(np.array(color1) - np.array(color2)) <= threshold)

# Load number templates
number_templates = {
    i: cv2.imread(resource_path(f"templates/number_{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(1, 6)
}

# Function to restart the process
def restart_process():
    global restart_flag
    restart_flag = True

# Create the main Tkinter window
root = tk.Tk()
root.title("Minesweeper Solver Control")
root.attributes('-topmost', True)
restart_flag = False
main_loop_thread = None
level = 1

# Function to open wizard for user input
def open_wizard():
    global x, y, w, h, rows, columns, restart_flag, main_loop_thread, level
    restart_flag = True
    if main_loop_thread is not None:
        main_loop_thread.join()  # Wait for the previous thread to finish

    cv2.destroyAllWindows()

    # Take a fullscreen screenshot and automatically detect the grid area using color detection
    screenshot = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    grid_mask = np.zeros(screenshot.shape[:2], dtype=np.uint8)

    # Convert screenshot to HSV for better color detection
    hsv_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])  # Lower bound for green color in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound for green color in HSV
    mask = cv2.inRange(hsv_screenshot, lower_green, upper_green)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Determine the top-left and bottom-right points of the grid
    if len(contours) > 0:
        grid_contour = contours[0]
        x, y, w, h = cv2.boundingRect(grid_contour)

        # Draw the detected grid area for preview
        preview_image = screenshot.copy()
        cv2.rectangle(preview_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Grid Detection Preview", preview_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Set rows and columns based on the current level
        level_dimensions = {
            1: (9, 9),
            2: (11, 12),
            3: (13, 15),
            4: (14, 18),
            5: (16, 20),
        }
        rows, columns = level_dimensions.get(level, (9, 9))
    else:
        print("Grid area could not be detected.")
        return

    restart_flag = False
    main_loop_thread = threading.Thread(target=start_main_loop)
    main_loop_thread.start()

# Function to go to the next level
def next_level():
    global level
    level += 1
    update_level_label()
    open_wizard()

# Function to reset to level 1
def reset_level():
    global level
    level = 1
    update_level_label()
    open_wizard()

# Function to quit the application
def quit_application():
    global restart_flag
    restart_flag = True
    root.quit()
    root.destroy()

# Function to update the level label
def update_level_label():
    level_label.config(text=f"Current Level: {level}")

# Function to update the threshold label
def update_threshold_label():
    threshold_label.config(text=f"Current Threshold: {threshold}")

# Function to set the threshold value
def set_threshold():
    global threshold
    try:
        new_threshold = int(threshold_input.get())
        threshold = new_threshold
        update_threshold_label()
    except ValueError:
        print("Invalid threshold value. Please enter an integer.")

# Create UI elements for the wizard
level_label = tk.Label(root, text=f"Current Level: {level}")
level_label.pack(pady=5)

info_label = tk.Label(root, text=f"On grid preview window (the blue lines one, press enter, if it doesnt align with your grid.\nThe grid needs to be completely untouched when you click start level/next level)")
info_label.pack(pady=5)

start_button = tk.Button(root, text="Start", command=open_wizard)
start_button.pack(pady=5)

next_level_button = tk.Button(root, text="Next Level", command=next_level)
next_level_button.pack(pady=5)

info_label = tk.Label(root, text=f"When you Start a new level, wait for the grid to show, then press Start level/Next level\nWe need the grid to initialize before we try detect the cells")
info_label.pack(pady=5)

reset_button = tk.Button(root, text="Reset to Level 1", command=reset_level)
reset_button.pack(pady=5)

# Create UI elements for threshold control
threshold_label = tk.Label(root, text=f"Current Threshold: {threshold}")
threshold_label.pack(pady=5)

threshold_info_label = tk.Label(root, text=f"How much an empty cell color can vary from the reference we have for it, small value=more strict")
threshold_info_label.pack(pady=5)

threshold_input = tk.Entry(root)
threshold_input.pack(pady=5)

set_threshold_button = tk.Button(root, text="Set Threshold", command=set_threshold)
set_threshold_button.pack(pady=5)

info_label_faq = tk.Label(root, text=f"If there are no green cells:\nUse echo slash illusion on a cell that you want to expand into,\nsometimes the algorithm cant figure it out")
info_label_faq.pack(pady=5)

quit_button = tk.Button(root, text="Quit", command=quit_application)
quit_button.pack(pady=5)

# Function to start the main loop
def start_main_loop():
    global restart_flag, rows, columns
    padding = 0

    while not restart_flag:
        screenshot = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        selected_region = screenshot[y:y + h, x:x + w].copy()
        cell_height = (h - (rows + 1) * padding) / rows
        cell_width = (w - (columns + 1) * padding) / columns

        # Draw the grid lines if debug mode is enabled
        if debug:
            for i in range(1, rows):
                y_pos = int(i * (cell_height + padding) + padding)
                cv2.line(selected_region, (0, y_pos), (w, y_pos), (255, 255, 255), 1)
            for j in range(1, columns):
                x_pos = int(j * (cell_width + padding) + padding)
                cv2.line(selected_region, (x_pos, 0), (x_pos, h), (255, 255, 255), 1)

        # Get colors at the center of each cell
        cell_colors = [[selected_region[int(i * (cell_height + padding) + padding + cell_height / 2),
                                        int(j * (cell_width + padding) + padding + cell_width / 2)].tolist()
                        if int(i * (cell_height + padding) + padding + cell_height / 2) < selected_region.shape[0] and
                           int(j * (cell_width + padding) + padding + cell_width / 2) < selected_region.shape[1]
                        else [0, 0, 0]
                        for j in range(columns)] for i in range(rows)]

        # Classify cells based on colors and template matching
        classified_grid = []
        for i in range(rows):
            classified_row = []
            for j in range(columns):
                top_left_x = int(j * (cell_width + padding) + padding)
                top_left_y = int(i * (cell_height + padding) + padding)
                bottom_right_x = int(top_left_x + cell_width)
                bottom_right_y = int(top_left_y + cell_height)

                if top_left_y < selected_region.shape[0] and bottom_right_y <= selected_region.shape[0] and \
                        top_left_x < selected_region.shape[1] and bottom_right_x <= selected_region.shape[1]:
                    cell_roi = selected_region[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                    gray_cell = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
                    found_number = False

                    for number, template in number_templates.items():
                        if template is None:
                            continue
                        for scale in np.linspace(0.7, 1.3, 7):
                            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                            if gray_cell.shape[0] >= resized_template.shape[0] and gray_cell.shape[1] >= resized_template.shape[1]:
                                res = cv2.matchTemplate(gray_cell, resized_template, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, _ = cv2.minMaxLoc(res)
                                if max_val > 0.85:  # Increased threshold for higher confidence
                                    classified_row.append(f"number_{number}")
                                    found_number = True
                                    break
                        if found_number:
                            break

                    if not found_number:
                        color = cell_colors[i][j]
                        if color_within_threshold(color, empty_cell_color, threshold):
                            classified_row.append("E")
                        else:
                            classified_row.append("unknown")
                else:
                    classified_row.append("unknown")
            classified_grid.append(classified_row)

        # Solve the grid using a more robust approach
        processed_cells = set()
        for _ in range(3):  # Run the processing multiple times to ensure all changes are accounted for
            for i in range(rows):
                for j in range(columns):
                    if (i, j) in processed_cells:
                        continue
                    if classified_grid[i][j].startswith("number_"):
                        number = int(classified_grid[i][j].split("_")[1])
                        neighbors = [(ni, nj) for ni in range(max(0, i - 1), min(rows, i + 2))
                                     for nj in range(max(0, j - 1), min(columns, j + 2)) if (ni, nj) != (i, j)]

                        unknown_neighbors = [(ni, nj) for ni, nj in neighbors if classified_grid[ni][nj] == "unknown"]
                        potential_bomb_count = sum(1 for ni, nj in neighbors if classified_grid[ni][nj] == "potential_bomb")

                        # If the number of potential bombs and unknowns matches the number, mark all unknowns as bombs
                        if len(unknown_neighbors) + potential_bomb_count == number:
                            for ni, nj in unknown_neighbors:
                                classified_grid[ni][nj] = "potential_bomb"
                                processed_cells.add((ni, nj))

                        # If the number of potential bombs matches the number, mark all unknowns as safe
                        elif potential_bomb_count == number:
                            for ni, nj in unknown_neighbors:
                                if classified_grid[ni][nj] == "unknown":
                                    classified_grid[ni][nj] = "safe"
                                    processed_cells.add((ni, nj))

                    processed_cells.add((i, j))

        # Re-evaluate the safety of cells based on a stricter confidence threshold
        safe_cells_found = False
        for i in range(rows):
            for j in range(columns):
                if classified_grid[i][j] == "unknown":
                    neighbors = [(ni, nj) for ni in range(max(0, i - 1), min(rows, i + 2))
                                 for nj in range(max(0, j - 1), min(columns, j + 2)) if (ni, nj) != (i, j)]
                    bomb_count = sum(1 for ni, nj in neighbors if classified_grid[ni][nj] == "potential_bomb")
                    number_neighbors = [classified_grid[ni][nj] for ni, nj in neighbors if classified_grid[ni][nj].startswith("number_")]

                    # Only mark as safe if we are almost certain based on all number neighbors
                    if len(number_neighbors) > 0:
                        inferred_bomb_count = sum(int(n.split("_")[1]) for n in number_neighbors) - bomb_count
                        if inferred_bomb_count == 0:
                            classified_grid[i][j] = "safe"
                            safe_cells_found = True

        # If no safe cells were found, re-evaluate the entire grid with additional constraints
        if not safe_cells_found:
            for i in range(rows):
                for j in range(columns):
                    if classified_grid[i][j] == "unknown":
                        neighbors = [(ni, nj) for ni in range(max(0, i - 1), min(rows, i + 2))
                                     for nj in range(max(0, j - 1), min(columns, j + 2)) if (ni, nj) != (i, j)]
                        bomb_count = sum(1 for ni, nj in neighbors if classified_grid[ni][nj] == "potential_bomb")
                        number_neighbors = [classified_grid[ni][nj] for ni, nj in neighbors if classified_grid[ni][nj].startswith("number_")]

                        # Re-evaluate and mark as safe if conditions are met
                        if len(number_neighbors) > 0:
                            inferred_bomb_count = sum(int(n.split("_")[1]) for n in number_neighbors) - bomb_count
                            if inferred_bomb_count == 0:
                                classified_grid[i][j] = "safe"

        # Additional check to ensure no missed safe cells
        for i in range(rows):
            for j in range(columns):
                if classified_grid[i][j].startswith("number_"):
                    number = int(classified_grid[i][j].split("_")[1])
                    neighbors = [(ni, nj) for ni in range(max(0, i - 1), min(rows, i + 2))
                                 for nj in range(max(0, j - 1), min(columns, j + 2)) if (ni, nj) != (i, j)]

                    unknown_neighbors = [(ni, nj) for ni, nj in neighbors if classified_grid[ni][nj] == "unknown"]
                    potential_bomb_count = sum(1 for ni, nj in neighbors if classified_grid[ni][nj] == "potential_bomb")

                    # If the number of potential bombs matches the number and there are unknowns, mark them as safe
                    if potential_bomb_count == number and len(unknown_neighbors) > 0:
                        for ni, nj in unknown_neighbors:
                            classified_grid[ni][nj] = "safe"

        # Draw the results on the region
        for i in range(rows):
            for j in range(columns):
                top_left_x = int(j * (cell_width + padding) + padding)
                top_left_y = int(i * (cell_height + padding) + padding)
                bottom_right_x = int(top_left_x + cell_width)
                bottom_right_y = int(top_left_y + cell_height)

                if debug and classified_grid[i][j].startswith("number_"):
                    number = classified_grid[i][j].split("_")[1]
                    cv2.putText(selected_region, str(number), (top_left_x + 5, top_left_y + int(cell_height) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if classified_grid[i][j] == "potential_bomb":
                    cv2.rectangle(selected_region, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 0), -1)
                elif classified_grid[i][j] == "safe":
                    cv2.rectangle(selected_region, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)
                    # Auto click if spacebar is held
                    if keyboard.is_pressed('space'):
                        click_x = x + top_left_x + cell_width // 2
                        click_y = y + top_left_y + cell_height // 2
                        pyautogui.click(click_x, click_y)
                elif debug and classified_grid[i][j] == "E":
                    cv2.putText(selected_region, "E", (top_left_x + 5, top_left_y + int(cell_height) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the selected region with the grid overlayed
        cv2.namedWindow("Selected Region with Grid", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Selected Region with Grid", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("Selected Region with Grid", w, h)
        cv2.imshow("Selected Region with Grid", selected_region)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep for a short duration to avoid excessive CPU usage
        time.sleep(0.1)

    cv2.destroyAllWindows()

# Run the Tkinter main loop
root.mainloop()
