import cv2
import numpy as np

# Pfad zum Video
video_path = "video.mov"

# Öffnen des Videos
video_capture = cv2.VideoCapture(video_path)

# Variable für das kombinierte Bild
combined_image = None
alpha = 0.1
# Anzahl der Frames im Video
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Durchlaufen der Frames
for frame_index in range(total_frames):
    # Extrahieren des aktuellen Frames
    ret, frame = video_capture.read()

    # Überprüfen, ob das Bild erfolgreich extrahiert wurde
    if ret:
        # Skalieren des Frames auf den Bereich von 0 bis 1
        frame = frame.astype(np.float32) / 255.0

        # Hinzufügen des aktuellen Frames zum kombinierten Bild
        if combined_image is None:
            combined_image = frame
        else:
            combined_image = alpha * frame + (1 - alpha) * combined_image
            #combined_image += frame
# Durchschnittsbild berechnen
#combined_image /= total_frames

# Skalieren des kombinierten Bildes auf den Bereich von 0 bis 255
combined_image = (combined_image * 255).astype(np.uint8)

# Speichern des kombinierten Bildes
cv2.imwrite("new.jpg", combined_image)

# Freigeben der Ressourcen
video_capture.release()
