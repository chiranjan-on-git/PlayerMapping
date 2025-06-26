# setup_homography.py

from src.utils import get_mouse_click_coords, calculate_and_save_homography

# --- STEP 1: Get coordinates from the TACTICAM video (our "destination" or "map" view) ---
# We will use frame 0 of the tacticam video.
# Run this script, and a window will pop up.
# Click on 4 distinct points on the field (e.g., corners of the penalty box).
# The script will print the coordinates in the terminal.
# Write those coordinates down in the 'tacticam_points' list below.
print("--- Getting points from Tacticam Video ---")
get_mouse_click_coords('data/tacticam.mp4', frame_number=109)

# --- STEP 2: Get coordinates from the BROADCAST video (our "source" or "angled" view) ---
# Now do the same for the broadcast video. Find the EXACT SAME points.
# You might need to change the frame_number to find a good, clear view.
print("\n--- Getting points from Broadcast Video ---")
get_mouse_click_coords('data/broadcast.mp4', frame_number=39)


# --- STEP 3: Fill in the coordinates and calculate the matrix ---
# After running the script once and getting your points, fill them in here.
# Make sure the order of points is the same in both lists!
# For example:
# Point 1 (e.g., top-left corner of penalty box) in tacticam_points must match
# Point 1 (top-left corner of penalty box) in broadcast_points.

# FILL THESE IN WITH THE COORDINATES YOU FOUND
broadcast_points = [
    (1453, 538), # Point 1
    (460, 898), # Point 2
    (201, 718), # Point 3
    (724, 576)  # Point 4
    # You can add more than 4 points for better accuracy
]

tacticam_points = [
    (694, 249), # Point 1
    (324, 652), # Point 2
    (118, 489), # Point 3
    (334, 315)  # Point 4
]

# Once the points are filled in, uncomment the line below and run the script again.
calculate_and_save_homography(broadcast_points, tacticam_points)

print("\nScript finished. Fill in the points and uncomment the final line to generate the matrix.")