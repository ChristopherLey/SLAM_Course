
def parse_landmarks(filename: str) -> list:
    """
    Parse the landmarks from the given file
    """
    landmarks = []
    with open(filename, 'r') as f:
        for line in f:
            landmarks.append([float(x) for x in line.strip().split()])
    return landmarks


class SensorData:
    def __init__(self, sensor_data: list):
        self.sensor_data = []
        self.odometry = []
        for line in sensor_data:
            if 'ODOMETRY' in line:
                self.odometry = [float(x) for x in line[len('ODOMETRY'):].strip().split()]
            elif 'SENSOR' in line:
                sensor = [float(s) for s in line[len('SENSOR'):].strip().split()]
                self.sensor_data.append(sensor)


def parse_sensor_data(filename: str) -> list:
    """
    Parse the sensor data from the given file
    """
    sensor_data = []
    data_buffer = []
    with open(filename, 'r') as f:
        for line in f:
            if 'ODOMETRY' in line:
                if len(data_buffer) > 0:
                    sensor_data.append(SensorData(data_buffer))
                data_buffer = [line, ]
            elif 'SENSOR' in line:
                data_buffer.append(line)
    if len(data_buffer) > 0:
        sensor_data.append(SensorData(data_buffer))
    return sensor_data
