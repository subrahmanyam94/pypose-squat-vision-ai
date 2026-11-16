"""MediaPipe 33-point landmark indices used by the app (no heavy imports)."""
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Skeleton lines for drawing (pairs of landmark indices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # arms + shoulders
    (11, 23), (12, 24), (23, 24),                              # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                    # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),  # feet
    (0, 11), (0, 12),                                          # head to shoulders (approx.)
]
