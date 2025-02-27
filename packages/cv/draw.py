#### random color
import random

def generate_random_color(id):
    random.seed(id)
    # Generate random light color (higher brightness values)
    r = random.randint(0, 255)  # Red component
    g = random.randint(0, 255)  # Green component
    b = random.randint(0, 255)  # Blue component
    return (r, g, b)

#### draw
#pip install opencv-python
import cv2

#cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3) #(0,255,0), 3) green
#cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Add ID label above the rectangle
#cv2.circle(frame, (x, y), 5, color, -1)  # Draw a circle at the current position
#cv2.line(frame, (prev_x, prev_y), (x, y), color, 3)  # Draw a line between previous and current position with regular line thickness