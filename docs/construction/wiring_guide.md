# Wiring Guide

## I. Soldering Guide

### 1. STM32 SVEA-PX4-Autopilot

Use a 90-degree pin header for this connection. Take a clear photo of the back side of the board, before soldering the pin header onto it. Make sure that the pins are oriented toward the inside of the PCB.

### 2. ZED Box power cable

Cut the thick original power cable. Strip the black outer sheath to expose the inner red and black wires, then strip the red and black insulation to expose the bare metal wire. Solder these to the white cable and wrap the joint with insulation tape. To identify the polarity of the white cable, hold it with the opening facing you and the latch on top: the left wire is positive and the right one is negative.

### 3. Lidar

Take the brown (+) and blue (−) wires and connect them to the white cable using the same method described in the ZED Box power cable section. The remaining thin wires are unused and can be left unconnected.

## II. Cable Connections

### 1. STM32 SVEA-PX4-Autopilot and Power Management Board(PMB)

Mount the Click board onto the STM32 board. Use the Mouser cable to connect the Click board to the Power Management Board (PMB). Then use two Dupont cables to connect the INT and GND pins located behind the USB Type-C port on the PMB to the EXT and GND pins on the STM32 board.

### 2. Chassis and PMB

The chassis should provide five bundles of servo cables plus the power cable of the ESC (Electronic Speed Controller), six cables in total. The five servo cables correspond, respectively, to steering (Steer), the ESC signal (throttle), the shift servo (gear), the front T-Lock (Fdiff) and the rear T-Lock (Rdiff). Label them with small stickers while building the chassis so that they stay identifiable. Connect the TRX-to-XT60 adapter to the ESC power cable. Route all six cables to the rear left side of the chassis and tie them together, then plug each one into the left side of the PMB; the corresponding ports are clearly marked on the board. Finally, use three short(15cm) female-to-female Dupont cables to connect the sbus 3V port on the PMB to the 7th row of the receiver mounted on the rear left side of the chassis.

### 3. Lidar and ZED Box

Connect the Ethernet cable of the lidar to the ZED Box, and its power cable to the front side of the PMB.

Connect the ZED Box power cable to the front side of the PMB as well. Then use a USB-A to Type-C cable to connect one of the USB ports on the ZED Box to the Type-C port on the STM32 board (the left one, marked PWR/USB).

### 4. Encoder and ESP32

KTH provides a custom connector cable for the encoders and the ESP32. It splits the power line into three branches and provides two signal lines. For power, use the 3.3V and GND pins on the STM32 board to supply the two encoders and the ESP32. For the signal lines, connect PB6 on the STM32 board to D7 on the ESP32, and PB7 to D6. Between the ESP32 and the encoders, connect the rear right encoder to pin D5 and the rear left encoder to pin D4 on the ESP32.

!!! tip

    Some cables run into the box, so a connector can be pulled loose later without anyone noticing. Once everything is connected, tape the cables to the boards so they stay in place.

### 5. Battery

Connect the balance lead (the white JST-XH connector) to the PMB, so that the board can monitor the individual cell voltages. Then fit a Deans-T-to-XT60 adapter to the battery's discharge lead and connect that to the PMB as well.

!!! warning

    The vehicle is powered as soon as the discharge lead is plugged in. Connect it last, once the rest of the wiring is complete and checked.


## III. Remote Control Settings

### 1. RX Low-Power Warning

Press Exit and Enter simultaneously, scroll down to Alarm, and set RX to 3 V.

### 2. Enable Connection to the Chassis

Take out the receiver and briefly press the button between its two antennas once. The blue light should turn on. Press and hold SWD to arm or disarm the chassis. Arming enables control of the chassis. Press and hold ESC until the red light blinks five times.

### 3. Reverse Throttle

When you run the car for the first time, the forward and reverse directions are reversed. To correct this, press Exit and Enter simultaneously, scroll down to Reverse, and set TH to REV.

### 4. Mode Switch

Use SWA to switch between high and low gear. SWB selects the control mode: back enables remote and program control, middle enables remote control only, and front disables all control.
