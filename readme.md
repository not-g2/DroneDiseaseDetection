# UAV Exploration Framework

## Overview

This project implements an autonomous UAV exploration system using **AirSim**.  
The drone explores an unknown area, avoids obstacles using **LiDAR**, and plans optimal paths using **A\***.  
It generates visualizations and coverage metrics to evaluate exploration performance.

## Requirements

-   **Python 3.9.11**

-   **Unreal 4.27.2**

-   **AirSim Plugin:** Latest version from [Microsoft AirSim GitHub Repository](https://github.com/microsoft/AirSim)

## Required Python Modules

Install all dependencies:

```bash
pip install numpy airsim matplotlib
```

## Steps

-   Move settings.json to AirSim Folder

```bash
Documents/AirSim/settings.json
```

-   Open Unreal Engine and open _any_ Environment compatible with AirSim (eg: Blocks)
-   Click Play
-   Run main.py

## Results

-   Exploration Results will be printed to the main console.
-   Intermediate Coverage map and final map will be found in images folder.
