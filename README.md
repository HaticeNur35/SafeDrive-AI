## SafeDrive AI ##

Check thermal_ai_control_v1_legacy.py for the initial threshold-based logic and thermal_ai_control_v2_optimized.py for the current PyTorch implementation.
An AI-driven motor protection system developed with **PyTorch**. This project is designed to enhance the safety and longevity of industrial motors by implementing intelligent control logic.

>Key Features
Gradual Throttle Reduction:Unlike traditional emergency stops, this system implements a multi-stage throttle reduction based on real-time sensor data.
Smart Monitoring:Continuously analyzes thermal and vibration levels to prevent hardware failure.
Autonomous Safety:Optimized for low-cost COTS (Commercial Off-The-Shelf) components, making it accessible for diverse engineering applications.

>Technical Overview
The model is trained to process multiple sensor inputs and determine the safest operating state for the motor.
Framework:PyTorch
Language:Python
Logic:The system calculates a safety factor (/S/) based on thermal input (/T/) and vibration magnitude (/V/): /S/ = /f(T, V)/

>Multi-Stage Safety Protocol:
Stage 1 (Warning):%20 Reduction when initial thresholds are met.
Stage 2 (Critical):%50 Reduction to prevent immediate damage.
Stage 3 (Emergency):Full Cut-off to ensure total hardware safety.

**Model Evolution & OptimizationThe development of SafeDrive-AI**
involved significant iterative improvements to ensure reliability and prevent system "panic" during operations.
>*Baseline (148 lines of code):* The initial version relied on rigid temperature thresholds (75°C, 85°C, 90°C without intermediate parameter processing.
>*The "Panic" Phenomenon:* At a high Mean Squared Error (MSE) of 0.169, the model was unstable—occasionally triggering an emergency shutdown at just 40°C while failing to react appropriately at 95°C.
>*Optimization Results:* By refining the PyTorch architecture and training parameters, the error rate was reduced from 0.169 to 0.0046.Current Reliability: The model now utilizes the parameter to maintain a smooth control curve, eliminating premature shutdowns and ensuring precise motor protection.


