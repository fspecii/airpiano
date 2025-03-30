# Air Piano

![Air Piano](https://img.shields.io/badge/Air%20Piano-Play%20Music%20in%20the%20Air-blue)
![Python](https://img.shields.io/badge/Python-3.7+-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-red)

Play the piano in thin air using just your webcam and hand movements! Air Piano is a virtual piano keyboard that uses computer vision to detect your hand positions and lets you play music without touching any physical keys.

## Features

- 🎹 **Virtual 3D Piano**: Fully rendered piano keyboard with perspective and visual feedback
- 👋 **Hand Tracking**: Advanced hand landmark detection using MediaPipe
- 🎵 **Real MIDI Sound**: Play actual piano notes through your computer's MIDI synthesizer
- 🎛️ **Customizable Layout**: Adjust the piano position, size, and orientation to fit your setup
- ⚙️ **Multiple Modes**: Position, calibrate, and play - easy to set up for your environment
- 💾 **Save Presets**: Store your preferred piano layout for future sessions

## How It Works

Air Piano uses your webcam to track your hand movements in real-time. The application displays a virtual keyboard on screen that you can "press" by moving your fingertips into the key areas. Each detected press triggers a MIDI note, creating realistic piano sounds.

The system uses:
- **OpenCV** for image processing and display
- **MediaPipe** for hand landmark detection
- **Pygame** for MIDI output and audio generation

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
- Pygame

Install dependencies:
```bash
pip install opencv-python mediapipe numpy pygame
```

## Usage

1. Run the application:
```bash
python main.py
```

2. **Positioning Mode** (Initial screen):
   - Use W/S/A/D keys to move the piano
   - Keys 1-6 adjust size and octaves
   - Z/X to flip orientation
   - Press C or Enter to continue

3. **Calibration Mode**:
   - Hold your hand steady over the keyboard for 3 seconds
   - System will calibrate to your hand position

4. **Playing Mode**:
   - Move your fingertips over the keys to play notes
   - Multiple notes can be played simultaneously
   - Visual feedback shows detected presses

## Controls

### Positioning Mode
- **W/S or ↑/↓**: Move piano up/down
- **A/D or ←/→**: Move piano left/right
- **Z/X**: Flip piano orientation
- **1/2**: Decrease/Increase width
- **3/4**: Decrease/Increase height
- **5/6**: Decrease/Increase octaves
- **H/V**: Toggle horizontal/vertical camera flip
- **F**: Save layout preset
- **L**: Load layout preset
- **C or Enter**: Continue to calibration

### Playing Mode
- **P**: Return to positioning mode
- **R**: Recalibrate
- **T**: Test sound
- **ESC**: Quit

## MIDI Output

The application uses MIDI for sound output. On Windows, it uses the Microsoft GS Wavetable Synth by default. For better sound quality, consider installing:
- **VirtualMIDISynth** (Windows)
- **FluidSynth** (macOS/Linux)
- Any software MIDI synthesizer with a SoundFont

## Troubleshooting

- **No sound?** Ensure your MIDI output device is working and press 'T' to test
- **Hands not detected?** Adjust lighting conditions and ensure your hands are clearly visible
- **Poor performance?** Lower the resolution in the code or use a computer with better specs

## Created by

Vali Neagu ([@AmbsdOP](https://x.com/AmbsdOP))
