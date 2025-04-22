# Hand Gesture Control System

## Features

- **Media Control**: Play/pause and skip tracks
- **Volume Control**: Adjust system volume up/down
- **Mouse Control**: Click and move operations
- **Scrolling**: Scroll up and down on pages
- **Windows Key**: Press the Windows key to open Start menu
- **Keyboard Typing**: Type letters using hand gestures

## Setup

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
python gesture_control.py
```

## Gesture Guide

### System Control Mode

| Gesture                                    | Action               |
| ------------------------------------------ | -------------------- |
| All fingers extended                       | Open Chrome          |
| Index & middle finger extended             | Open Notepad         |
| Only thumb extended                        | Play/Pause media     |
| Thumb & index extended                     | Next track           |
| Thumb, index & pinky extended              | Press Windows key    |
| Index, middle & ring extended              | Volume up            |
| Index & pinky extended                     | Volume down          |
| Only index finger pointed                  | Mouse click          |
| All fingers extended upward                | Scroll up            |
| All fingers pointing downward              | Scroll down          |
| Thumb & index pinched with others extended | Toggle keyboard mode |

### Keyboard Mode

| Gesture                                    | Key                                |
| ------------------------------------------ | ---------------------------------- |
| All fingers extended                       | A                                  |
| Index & middle finger extended             | B                                  |
| Only thumb extended                        | C                                  |
| Thumb & index extended                     | D                                  |
| Flat palm with fingers together            | Space                              |
| Swipe left                                 | Backspace                          |
| Thumb & index pinched with others extended | Toggle back to system control mode |

## Requirements

- Python 3.7+
- Webcam
- Decent lighting for accurate hand detection

## Tips

- Toggle between System Control Mode and Keyboard Mode to access different functions
- Keep your hand steady and within frame for best detection results
- Ensure good lighting conditions for accurate gesture recognition
