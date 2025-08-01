from neurosdk.scanner import Scanner
from em_st_artifacts.utils import lib_settings
from em_st_artifacts.utils import support_classes
from em_st_artifacts import emotional_math
from neurosdk.cmn_types import *

from time import sleep
import csv
from datetime import datetime

# Open CSV log file
csv_file = open('brain_data_log.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)
'''csv_writer.writerow([
    'timestamp', 'calibration_percent', 'is_artifacted_sequence',
    'mental_param_1', 'mental_param_2', 'mental_param_3',
    'spectral_delta', 'spectral_theta', 'spectral_alpha', 'spectral_beta', 'spectral_gamma'
])'''

def sensor_found(scanner, sensors):
    for index in range(len(sensors)):
        print('Sensor found: %s' % sensors[index])

def on_sensor_state_changed(sensor, state):
    print('Sensor {0} is {1}'.format(sensor.name, state))

def on_battery_changed(sensor, battery):
    print('Battery: {0}'.format(battery))

def classify_attention(calib, is_artifact, m1, m2, m3, delta, theta, alpha, beta, gamma):
    if alpha >= 0.52 and theta < 0.21 and beta >= 0.26:
        return "Focused"
    elif alpha >= 0.52 and beta < 0.26:
        return "Relaxed"
    elif theta >= 0.21 and beta < 0.26:
        return "Drowsy"
    else:
        return "Neutral"
def on_signal_received(sensor, data):
    raw_channels = []
    for sample in data:
        left_bipolar = sample.T3 - sample.O1
        right_bipolar = sample.T4 - sample.O2
        raw_channels.append(support_classes.RawChannels(left_bipolar, right_bipolar))

    math.push_data(raw_channels)
    math.process_data_arr()

    if not math.calibration_finished():
        calibration_percent = math.get_calibration_percents()
    else:
        mental_data = math.read_mental_data_arr()
        spectral_data = math.read_spectral_data_percents_arr()
        is_artifact = math.is_artifacted_sequence()

        if mental_data and spectral_data:
            md = mental_data[0]
            sd = spectral_data[0]

            # Safely extract mental params
            if hasattr(md, 'get_data'):
                mental_values = md.get_data()
            else:
                mental_values = list(md.__dict__.values())

            # Unpack values for clarity
            calibration_percent = 100
            is_artifacted = int(is_artifact)
            m1, m2, m3 = mental_values[:3]
            delta, theta, alpha, beta, gamma = sd.delta, sd.theta, sd.alpha, sd.beta, sd.gamma

            # Classify attention state
            attention_state = classify_attention(
                calibration_percent, is_artifacted, m1, m2, m3, delta, theta, alpha, beta, gamma
            )

            # Compose row
            timestamp = datetime.now().isoformat()
            row = [
                timestamp,
                calibration_percent,
                is_artifacted,
                m1, m2, m3,
                delta, theta, alpha, beta, gamma,
                attention_state
            ]

            # Write to CSV
            csv_writer.writerow(row)
            print("Logged data:", row)


def on_resist_received(sensor, data):
    print("O1 resist is normal: {0}. Current O1 resist {1}".format(data.O1 < 2000000, data.O1))
    print("O2 resist is normal: {0}. Current O2 resist {1}".format(data.O2 < 2000000, data.O2))
    print("T3 resist is normal: {0}. Current T3 resist {1}".format(data.T3 < 2000000, data.T3))
    print("T4 resist is normal: {0}. Current T4 resist {1}".format(data.T4 < 2000000, data.T4))

try:
    scanner = Scanner([SensorFamily.LEHeadband])

    scanner.sensorsChanged = sensor_found
    scanner.start()
    print("Starting search for 25 sec...")
    sleep(25)
    scanner.stop()

    sensorsInfo = scanner.sensors()
    for i in range(len(sensorsInfo)):

        current_sensor_info = sensorsInfo[i]
        sensor = scanner.create_sensor(current_sensor_info)
        print("Current connected device {0}".format(current_sensor_info))

        sensor.sensorStateChanged = on_sensor_state_changed
        sensor.batteryChanged = on_battery_changed
        sensor.signalDataReceived = on_signal_received
        sensor.resistDataReceived = on_resist_received

        sensor.exec_command(SensorCommand.StartResist)
        print("Start resistance")
        sleep(20)
        sensor.exec_command(SensorCommand.StopResist)
        print("Stop resistance")

        calibration_length = 6
        nwins_skip_after_artifact = 10

        mls = lib_settings.MathLibSetting(sampling_rate=250,
                                          process_win_freq=25,
                                          n_first_sec_skipped=4,
                                          fft_window=1000,
                                          bipolar_mode=True,
                                          squared_spectrum=True,
                                          channels_number=4,
                                          channel_for_analysis=0)
        ads = lib_settings.ArtifactDetectSetting(art_bord=110,
                                                 allowed_percent_artpoints=70,
                                                 raw_betap_limit=800_000,
                                                 global_artwin_sec=4,
                                                 num_wins_for_quality_avg=125,
                                                 hamming_win_spectrum=True,
                                                 hanning_win_spectrum=False,
                                                 total_pow_border=400_000_000,
                                                 spect_art_by_totalp=True)
        sads = lib_settings.ShortArtifactDetectSetting(ampl_art_detect_win_size=200,
                                                       ampl_art_zerod_area=200,
                                                       ampl_art_extremum_border=25)
        mss = lib_settings.MentalAndSpectralSetting(n_sec_for_averaging=2,
                                                    n_sec_for_instant_estimation=4)

        math = emotional_math.EmotionalMath(mls, ads, sads, mss)
        math.set_calibration_length(calibration_length)
        math.set_mental_estimation_mode(False)
        math.set_skip_wins_after_artifact(nwins_skip_after_artifact)
        math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
        math.set_spect_normalization_by_bands_width(True)

        if sensor.is_supported_command(SensorCommand.StartSignal):
            sensor.exec_command(SensorCommand.StartSignal)
            print("Start signal")
            math.start_calibration()
            sleep(120)
            sensor.exec_command(SensorCommand.StopSignal)
            print("Stop signal")

        sensor.disconnect()
        print("Disconnect from sensor")
        del sensor
        del math

    del scanner
    print('Remove scanner')
except Exception as err:
    print(err)
finally:
    csv_file.close()
    print("CSV file closed.")