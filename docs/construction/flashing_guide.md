# Flashing Guide

## SVEA-PX4-Autopilot

刷机：
https://github.com/nilskiefer/SVEA-PX4-Autopilot#

right bottom connect reopen in container, wait for a while around 10 mins

install codegrip

connect your laptop to the top right type C port(with PWR/DBG), use lsusb or codegrip(scan device) to check if MikroElektronika MikroSDK TinyUSB Device is there.

Follow SVEA-PX4-Autopilot/docs/en/svea/build-and-flash.md

git config --global --add safe.directory '*'(no use if you click manage unsafe repository)

rm -rf build/mikroe_clicker4-stm32f7_noboot
make mikroe_clicker4-stm32f7_noboot

sudo apt install openocd

reopen vscode and reopen container

codegrip firmware update in windows!

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


# ESP for Encoder

https://github.com/nilskiefer/SVEA-PX4-Autopilot-MCU-Peripheral

follow the readme guide starting from Host Python venv