# Flashing Guide

## SVEA-PX4-Autopilot

### Steps
1. Install CODEGRIP on Windows and Linux.
2. Connect your laptop to the USB Type-C port on the right side of the STM32 board (labeled PWR/DBG). Run ```lsusb``` in the terminal or use CODEGRIP's device scan to check that MikroElektronika device appears.
3. Update the firmware using CODEGRIP on Windows(Linux doesn't work). Go to Help->Updates->CODEGRIP. Unplug your device, click SCAN DEVICES, and reconnect the device within 5 seconds. Once CODEGRIP detects the device, you can update the firmware.
4. Clone https://github.com/nilskiefer/SVEA-PX4-Autopilot# to a local folder and open it in VS Code. When the notification appears in the bottom-left corner, click "Reopen in Container" and select "PX4 detect" at the top.
5. Follow the guide in SVEA-PX4-Autopilot/docs/en/svea/build-and-flash.md and make sure openocd is installed. Close CODEGRIP before running the openocd command so that CODEGRIP does not keep the device busy.

### Notes
1. To rebuild, remove the existing build first:
```bash
rm -rf build/mikroe_clicker4-stm32f7_noboot
make mikroe_clicker4-stm32f7_noboot
```
2. If a notification appears in the bottom-right corner, click "Manage Unsafe Repository", then select "All Repositories" at the top. If you miss the notification, run the following command in the terminal instead:
```bash
git config --global --add safe.directory '*'
```

## ESP32

### Steps

1. Clone https://github.com/nilskiefer/SVEA-PX4-Autopilot-MCU-Peripheral to a local folder.
2. Follow the instructions in the README, starting with the "Host Python venv" section.

## ZED Box

How to flash Zed box,
1. connect OTG to your laptop before power on. Do not plug in PoE.
2. sudo dmesg -w
to check if there is “usb 3-6: Manufacturer: NVIDIA Corp”, which means it is in the recovery mode
3. on the website
https://docs.stereolabs.com/docs/products/embedded/zed-box-orin/reset-update#force-the-zed-box-in-recovery-mode
download NVIDIA® JetPack 6.2 / L4T 36.4 - JetPack details
4. in the terminal exit dmesg
cd ~/Downloads
chmod +x zedbox_all_usb_flash_6.2.sh
BSP_ROOT=./stereolabs_bsp/ ./zedbox_all_usb_flash_6.2.sh

5. in zedbox
fix wifi and browser issue: https://docs.stereolabs.com/docs/products/embedded/zed-box-orin/troubleshooting
git clone https://github.com/kth-sml/svea
sudo apt install curl
util/get-docker
sudo usermod -aG docker user

6. set up lidar,
docs/setup/lidar_setup
notice: one error in original: ip address should be 192.168.0.100

ping 192.168.0.10


