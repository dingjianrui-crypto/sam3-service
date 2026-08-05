# Requirement Anaylysis

let's investigate the sprint kayak paddle catch and exit event, the video is captured from the side view, the kayak can move in two different directions. and the paddle can be in clockwise rotation and anticlockwise rotation.  the event detection process as below:
1. detect the kayak movement direction based on the paddle rotation. it's moving towards left if paddle make anticlockwise rotation, it's moving towards right if the paddle makes clockwise rotation
2. the paddle catch event cannot be detected when the paddle stroke happened on the other side of the camera.
3. assume the kayak is moving towards left and the paddle is rotating anticlockwise direction, the paddle angle is calculated based on the detected waterline from the left side, and the angle start from the under waterline from 0 to 360 degree. within 360 degree the maximum number of catch and exit is 1.
4. assume tha kayak is moving towards right and the paddle is rotating clockwise direction, the paddle angle is calculated based on the detected waterline from right side, and the angle start from the under waterline from 0 to 360 degree.
5. catch and exit event reset automatically in the next phase even when some of the events are not detected.
6. catch event - the first time when paddle centerline and waterline are crossed in each phase.
7. exit event - the first time when the paddle centerline and waterline are seperated in each phase.