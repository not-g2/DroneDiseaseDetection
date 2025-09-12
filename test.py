from dronekit import connect
import time

# Connect to the vehicle (adjust connection string to your setup)
# For SITL: 'tcp:127.0.0.1:5760' or 'udp:127.0.0.1:14550'
connection_string = 'udp:127.0.0.1:14550'
vehicle = connect(connection_string, wait_ready=True)

print("Connected to vehicle. Reading distance sensor...")

try:
    while True:
        # DISTANCE_SENSOR is available in vehicle.rangefinder
        distance = vehicle.rangefinder.distance  # in meters
        voltage = vehicle.rangefinder.voltage    # sensor voltage, optional
        print(f"Distance: {distance:.2f} m, Voltage: {voltage:.2f} V")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    vehicle.close()
