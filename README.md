# Cross-Camera Player Mapping

## Objective
This project maps football players across two different video feeds—a wide-angle tacticam view and a standard broadcast view—ensuring each player is assigned a single, consistent, and long-term ID across both streams.

## Demo
![PlayerMapping](https://github.com/user-attachments/assets/7c7cc6de-154e-4ddf-827c-af074377f7a0)
A sample output showing the final mapping. Players in both views share the same global ID, presented in a clean, non-distorted side-by-side view.

## Key Features
- **Hybrid Matching Algorithm**: Employs a sophisticated cost function that combines both spatial distance (using homography) and visual appearance (using HSV color histograms) for highly accurate player matching.
- **Advanced Temporal Re-identification**: The system maintains a memory of players who are temporarily lost or occluded. It uses a history of their position and color signature to robustly re-assign the correct ID when they reappear.
- **Robust Detection Pipeline**: Raw detections from the YOLO model are post-processed using a custom Non-Maximum Suppression (NMS) filter to eliminate redundant, overlapping boxes, ensuring cleaner data for the tracking stage.
- **Professional & Modular Architecture**: The project is built with a clean separation of concerns, featuring dedicated classes for Detection, Tracking, and a highly advanced PlayerMapper that encapsulates the core re-identification logic.
- **Distortion-Free Visualization**: The final output displays both video feeds side-by-side without distorting their aspect ratios, providing a clear and professional demonstration of the results.

## Methodology & Approach
The core of my solution is a multi-stage pipeline designed for accuracy and robustness, going beyond simple geometric mapping.

1. **Detection & Filtering**: Players are first detected using the provided YOLO model. I then implemented a crucial post-processing step to filter these detections, removing overlapping bounding boxes based on an IoU (Intersection over Union) threshold. This ensures each player is represented by a single, high-confidence box.
2. **Homography Estimation**: A homography matrix serves as the geometric bridge between the two camera views. I developed a utility to select corresponding reference points and compute this matrix, which is then used to translate player positions from the broadcast view into the tacticam's top-down coordinate space.
3. **Intra-Video Tracking**: A Tracker class assigns a stable, temporary ID to each player within their own video. This handles minor, frame-to-frame detection gaps and provides a consistent input for the final mapping stage.
4. **Advanced Cross-Camera Mapping**: This is handled by a dedicated PlayerMapper class, which represents the core intelligence of the system. For each frame, it performs the following:
   - **Transformation**: It transforms the broadcast players' positions using the homography matrix.
   - **Feature Extraction**: It calculates a normalized HSV color histogram for every player in both views. This captures their visual appearance (e.g., jersey color).
   - **Cost Matrix Calculation**: It builds a cost matrix where the "cost" of matching two players is a weighted sum of their spatial distance and the dissimilarity of their color histograms.
   - **Optimal Assignment**: It uses the Hungarian algorithm to find the optimal global pairings that minimize the total cost.
   - **ID Persistence & Re-identification**: The mapper maintains a memory of recently lost players. If an unassigned player appears near the last known location of a lost player and has a similar color profile, the system intelligently reassigns the correct long-term ID.

## Technical Challenges & Solutions
This project presented several advanced challenges, which I solved with targeted engineering solutions:

### Challenge: Ambiguity from Spatially Close Players
**Problem**: A homography-only approach fails when multiple players are clustered together. Their transformed positions are too close to reliably distinguish them.

**Solution**: I solved this by implementing a hybrid cost function. By adding a player's color histogram (capturing jersey color) as a second feature, the system can easily differentiate between two nearby players (e.g., one in a white kit and one in a red kit), dramatically increasing mapping accuracy.

### Challenge: Maintaining Long-Term Identity Through Occlusions
**Problem**: A player might be occluded or leave the frame and reappear later. A simple tracker would assign them a new ID, breaking consistency.

**Solution**: I engineered the PlayerMapper with a temporal memory. It tracks the last known state (position and color signature) of every player with a final ID. This allows it to perform re-identification, correctly matching a returning player to their original ID, which is critical for long-term analytics.

### Challenge: Noisy and Redundant Detector Output
**Problem**: Object detection models can sometimes output multiple, slightly different bounding boxes for the same player. This "noise" can confuse the tracking and mapping algorithms.

**Solution**: I implemented a Non-Maximum Suppression (NMS) filter as a post-processing step. This function intelligently discards redundant, lower-confidence detections that have a high IoU with a higher-confidence detection, ensuring that each player is represented by one clean bounding box.

### Challenge: Professional and Clear Visualization
**Problem**: The two video feeds have different resolutions and aspect ratios. Simply resizing them to be the same height would distort the image, making the results look unprofessional.

**Solution**: I created a resize_and_pad utility function. It resizes each frame to fit within a target panel while maintaining its original aspect ratio, padding the empty space. This results in a clean, clear, and non-distorted side-by-side comparison.

## Project Structure
The code is organized in a modular way for clarity and maintainability.
**Please refer to [`Structure.txt`](Structure.txt)** for a complete view of the folder hierarchy and file organization.

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/chiranjan-on-git/PlayerMapping.git
cd PlayerMapping
```
### 2. Download the Model
Download the trained model from the following link:

🔗 [best.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

Once downloaded, create a folder named `models` in the root of the project (if it doesn't already exist), and place the file inside it with the name.
### 3. Set Up the Environment
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
python main.py
```
### 5. (Optional) Recalculate Homography
```bash
python setup_homography.py
```
