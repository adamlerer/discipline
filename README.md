# Discipline - Cat Food Bowl Guardian

A hardware/software system that ensures each cat eats only from their designated bowl. Uses computer vision to identify cats and triggers a water spray deterrent when the wrong cat eats from the wrong bowl.

## The Problem

Ilana keeps eating Abbi's food. This system watches the food bowls and sprays Ilana with water if she tries to eat from Abbi's bowl before Abbi walks away.

## How It Works

1. Camera monitors the feeding area continuously
2. When a cat approaches a bowl, the system identifies which cat it is (Abbi or Ilana)
3. Tracks which cat is at which bowl
4. If Ilana eats from Abbi's bowl while Abbi hasn't finished, activates the water sprayer
5. Logs all feeding events for analysis

## Hardware Required

### Core Components

| Component | Description | Approximate Cost |
|-----------|-------------|------------------|
| Raspberry Pi 4 (4GB+) | Main controller running the vision system | $55-75 |
| Raspberry Pi Camera Module 3 | Wide-angle camera for monitoring | $25-35 |
| 12V Solenoid Valve (normally closed) | Controls water flow | $10-15 |
| 5V Relay Module | Switches the solenoid valve | $5-8 |
| 12V Power Supply (2A) | Powers the solenoid | $10-15 |
| USB-C Power Supply (5V 3A) | Powers the Raspberry Pi | $10-15 |
| Silicone tubing (1/4" inner diameter) | Connects water source to nozzle | $8-12 |
| Spray nozzle | Creates a fine mist spray | $5-10 |
| Water reservoir (1-2 gallon) | Gravity-fed water source | $10-15 |
| Jumper wires | Connections | $5-8 |
| Project enclosure | Weatherproof box for electronics | $15-20 |

**Total estimated cost: $160-230**

### Recommended Products (Amazon/Adafruit)

1. **Raspberry Pi 4 Model B (4GB)** - Any official retailer
2. **Raspberry Pi Camera Module 3 Wide** - For better field of view
3. **DIGITEN 12V 1/4" Solenoid Valve** - Commonly available, food-safe
4. **HiLetgo 5V Relay Module** - Optoisolated for safety
5. **12V 2A DC Power Adapter** - Standard barrel jack
6. **Misting Nozzle Kit** - Garden misting nozzles work well

### Optional Enhancements

- **IR illuminator** - For nighttime operation
- **NoIR Camera Module** - Better low-light performance
- **Pressure tank** - For more consistent spray pressure
- **Flow sensor** - To monitor water usage

## Shopping List (with Purchase Links)

### Core Components

| Component | Best Price | Amazon Alternative |
|-----------|------------|-------------------|
| **Raspberry Pi 4 (4GB)** | [$60 - Adafruit](https://www.adafruit.com/product/4296) | [CanaKit Starter Kit - Amazon](https://www.amazon.com/CanaKit-Raspberry-4GB-Starter-Kit/dp/B07V5JTMV9) |
| **Pi Camera Module 3 Wide** | [$35 - Adafruit](https://www.adafruit.com/product/5658) | [Camera Module 3 Wide - Amazon](https://www.amazon.com/Raspberry-Pi-Camera-Module-Wide/dp/B0BRY757NX) |
| **12V Solenoid Valve (1/4" NC)** | - | [DIGITEN 12V Solenoid - Amazon](https://www.amazon.com/DIGITEN-Solenoid-Connect-normally-Closed/dp/B016MP1HX0) |
| **5V Relay Module (Opto-isolated)** | - | [HiLetgo 5V Relay - Amazon](https://www.amazon.com/HiLetgo-Channel-optocoupler-Support-Trigger/dp/B00LW15A4W) |
| **12V 2A Power Supply** | - | [12V 2A Adapter - Amazon](https://www.amazon.com/Coming-Data-5-5x2-1mm-Connector-Certified/dp/B0194B7WSI) |
| **USB-C Power Supply (5V 3A)** | - | [Raspberry Pi 4 Power Supply - Amazon](https://www.amazon.com/UCTRONICS-Raspberry-Supply-Charger-Adapter/dp/B08C9VYLLK) |
| **Silicone Tubing (1/4" ID)** | - | [Food Grade Silicone Tubing 10ft - Amazon](https://www.amazon.com/Pure-Silicone-Tubing-High-Kink-Free/dp/B07QDKJ66Q) |
| **Misting Nozzle Kit** | - | [Brass Misting Nozzles - Amazon](https://www.amazon.com/lifeegrn-Misting-Nozzles-Connectors-Outdoor/dp/B098F275FB) |
| **Jumper Wires** | - | [120pcs Jumper Wire Kit - Amazon](https://www.amazon.com/Jumper-Female-Wire-Arduino-raspberry/dp/B01MT530B8) |
| **Waterproof Enclosure** | [$9.95 - Adafruit](https://www.adafruit.com/product/3931) | [Sixfab IP65 Enclosure - Amazon](https://www.amazon.com/Outdoor-Enclosure-Raspberry-Development-Boards/dp/B09TRZ5BTB) |

### Quick Links

**Adafruit (often cheaper):**
- [Raspberry Pi 4 4GB - $60](https://www.adafruit.com/product/4296)
- [Camera Module 3 Wide - $35](https://www.adafruit.com/product/5658)
- [Camera Module 3 Wide NoIR (for low light) - $35](https://www.adafruit.com/product/5660)
- [Weatherproof Enclosure with Cable Glands](https://www.adafruit.com/product/3931)

**Amazon (all-in-one shopping):**
- [Raspberry Pi 4 4GB Board](https://www.amazon.com/Raspberry-Model-2019-Quad-Bluetooth/dp/B07TC2BK1X)
- [Pi Camera Module 3 Wide](https://www.amazon.com/Raspberry-Pi-Camera-Module-Wide/dp/B0BRY757NX)
- [DIGITEN 12V 1/4" Solenoid Valve](https://www.amazon.com/DIGITEN-Solenoid-Connect-normally-Closed/dp/B016MP1HX0)
- [HiLetgo 5V Relay Module (2-pack)](https://www.amazon.com/HiLetgo-Channel-optocoupler-Support-Trigger/dp/B00LW15A4W)
- [12V 2A DC Power Adapter (UL Certified)](https://www.amazon.com/Coming-Data-5-5x2-1mm-Connector-Certified/dp/B0194B7WSI)
- [USB-C 5V 3A Power Supply with Switch](https://www.amazon.com/UCTRONICS-Raspberry-Supply-Charger-Adapter/dp/B08C9VYLLK)
- [Pure Silicone Tubing 1/4" x 10ft](https://www.amazon.com/Pure-Silicone-Tubing-High-Kink-Free/dp/B07QDKJ66Q)
- [Brass Misting Nozzle Kit](https://www.amazon.com/lifeegrn-Misting-Nozzles-Connectors-Outdoor/dp/B098F275FB)
- [Jumper Wires Kit (120pcs M/F/M-M/F-F)](https://www.amazon.com/Jumper-Female-Wire-Arduino-raspberry/dp/B01MT530B8)
- [Sixfab IP65 Outdoor Enclosure](https://www.amazon.com/Outdoor-Enclosure-Raspberry-Development-Boards/dp/B09TRZ5BTB)

## Wiring Diagram

```
                                    +12V Power Supply
                                           |
                                           v
[Raspberry Pi 4]                    [12V Solenoid Valve]
      |                                    ^
      | GPIO 17                            |
      v                                    |
[5V Relay Module] -----> Switches --------+
      |
      | VCC -> Pi 5V
      | GND -> Pi GND
      | IN  -> Pi GPIO 17

[Pi Camera] -----> CSI Port on Raspberry Pi

Water Flow:
[Reservoir] --> [Tubing] --> [Solenoid Valve] --> [Tubing] --> [Spray Nozzle]
                                                                    |
                                                                    v
                                                            [Aimed at Abbi's bowl]
```

## Software Setup

### 1. Raspberry Pi OS Setup

```bash
# Flash Raspberry Pi OS (64-bit) to SD card using Raspberry Pi Imager
# Enable SSH and set up WiFi during imaging

# After first boot, update the system
sudo apt update && sudo apt upgrade -y

# Enable the camera
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable

# Install required system packages
sudo apt install -y python3-pip python3-venv libcamera-apps python3-libcamera python3-picamera2
```

### 2. Project Installation

```bash
# Clone the repository
git clone https://github.com/adamlerer/discipline.git
cd discipline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download the cat detection model (YOLOv8)
python scripts/download_model.py
```

### 3. Cat Training (Teaching the System Your Cats)

Before the system can identify Abbi and Ilana, you need to train it with photos of each cat.

```bash
# Capture training images for each cat
python scripts/capture_training_images.py --cat abbi --count 50
python scripts/capture_training_images.py --cat ilana --count 50

# Train the cat classifier
python scripts/train_classifier.py
```

### 4. Configuration

Edit `config.yaml` to set up your specific environment:

```yaml
# Bowl positions (in pixel coordinates from camera view)
bowls:
  abbi:
    x: 200
    y: 300
    radius: 80
  ilana:
    x: 500
    y: 300
    radius: 80

# Spray settings
spray:
  duration_ms: 500  # How long to spray
  cooldown_s: 10    # Minimum time between sprays

# Camera settings
camera:
  resolution: [1280, 720]
  framerate: 30
```

### 5. Calibration

```bash
# Run the calibration tool to set bowl positions
python scripts/calibrate_bowls.py
```

### 6. Running the System

```bash
# Start the discipline system
python -m discipline.main

# Or run as a service (recommended for production)
sudo cp discipline.service /etc/systemd/system/
sudo systemctl enable discipline
sudo systemctl start discipline
```

## Project Structure

```
discipline/
├── README.md
├── requirements.txt
├── config.yaml
├── discipline/
│   ├── __init__.py
│   ├── main.py              # Main entry point
│   ├── camera.py            # Camera capture and processing
│   ├── cat_detector.py      # YOLO-based cat detection
│   ├── cat_identifier.py    # Identifies Abbi vs Ilana
│   ├── bowl_monitor.py      # Tracks cats at bowls
│   ├── sprayer.py           # Controls water sprayer
│   └── logger.py            # Event logging
├── scripts/
│   ├── download_model.py
│   ├── capture_training_images.py
│   ├── train_classifier.py
│   └── calibrate_bowls.py
├── models/                   # Trained models stored here
├── data/
│   └── training/            # Training images
└── logs/                    # Event logs
```

## Safety Notes

1. **Water and Electronics**: Keep all electronics in a waterproof enclosure away from the spray zone
2. **Electrical Safety**: Use proper gauge wires and ensure all connections are secure
3. **Cat Safety**: The spray should be a gentle mist, not a powerful jet. Test pressure before deployment
4. **Supervision**: Monitor the system during initial deployment to ensure correct operation

## Troubleshooting

### Camera not detected
```bash
# Check camera connection
libcamera-hello --list-cameras
```

### Relay not triggering
```bash
# Test GPIO manually
python -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); GPIO.setup(17, GPIO.OUT); GPIO.output(17, True)"
```

### Cats not being identified correctly
- Ensure good lighting in the feeding area
- Capture more training images from various angles
- Check that cats are visually distinct (color, pattern, size)

## License

MIT License - Feel free to use and modify for your own cat discipline needs!
