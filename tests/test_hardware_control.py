"""Integration-style tests for IndiController using a fully mocked PyIndi stack."""

import unittest
from unittest.mock import MagicMock, patch
import threading
import time
import queue
import sys
import os
import tempfile

REAL_SLEEP = time.sleep  # Keep original sleep for test fakes

# Add the parent directory to the system path to allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Mock PyIndi library for testing
class MockPyIndi:
    B_CLIENT = 0 # Example value, actual value doesn't matter for mock
    ISS_OFF = 0
    ISS_ON = 1
    SR_1OFMANY = 1 # Switch rule
    _properties = {}  # Shared property store across mock devices

    class BaseClient:
        def __init__(self):
            # Share a single store across all BaseClient instances
            self._properties = MockPyIndi._properties
            self._blobs = {}
            self._blob_queue = queue.Queue() # For simulating blob arrival
            self.isConnected_val = False
            self._devices = {}

        def setServer(self, host, port): pass
        def connectServer(self, *args, **kwargs): 
            self.isConnected_val = True
            return True
        def disconnectServer(self): 
            self.isConnected_val = False

        def isConnected(self): return self.isConnected_val

        def getDevice(self, name):
            if name in self._devices:
                return self._devices[name]
            mock_dev = MagicMock()
            mock_dev.getDeviceName.return_value = name
            mock_dev.getNumber.side_effect = lambda prop_name: self.getNumber(name, prop_name)
            mock_dev.getSwitch.side_effect = lambda prop_name: self.getSwitch(name, prop_name)
            mock_dev.getBLOB.side_effect = lambda prop_name: self.getBLOB(name, prop_name)
            self._devices[name] = mock_dev
            return mock_dev
        
        def getDevices(self): return [] # Simplified

        def getNumber(self, device_name, prop_name):
            return self._properties.get(f"{device_name}.{prop_name}")
        def getSwitch(self, device_name, prop_name):
            return self._properties.get(f"{device_name}.{prop_name}")
        def getBLOB(self, device_name, prop_name):
            return self._properties.get(f"{device_name}.{prop_name}")

        def sendNewNumber(self, prop): 
            self._properties[f"{prop.getDeviceName()}.{prop.name}"] = prop
            # Simulate property state change
            prop.s = MockPyIndi.IPS_BUSY # Assume busy after sending command
            self.newProperty(prop) # Trigger newProperty callback

        def sendNewSwitch(self, prop):
            self._properties[f"{prop.getDeviceName()}.{prop.name}"] = prop
            self.newProperty(prop) # Trigger newProperty callback

        def setBLOBMode(self, mode, device_name, prop_name): pass

        # Callbacks (simulated)
        def newProperty(self, prop): pass # To be overridden by IndiController
        def newBLOB(self, bp): pass # To be overridden by IndiController
        
    IPS_IDLE = 0
    IPS_OK = 1
    IPS_BUSY = 2
    IPS_ALERT = 3

    class BaseDevice:
        # These methods are typically called by _wait_prop in hardware_control.py
        # They need to be callable, even if they just return None for the mock.
        def getNumber(self, name):
            dev_name = getattr(self, "getDeviceName", lambda: None)()
            return MockPyIndi._properties.get(f"{dev_name}.{name}")
        def getSwitch(self, name):
            dev_name = getattr(self, "getDeviceName", lambda: None)()
            return MockPyIndi._properties.get(f"{dev_name}.{name}")
        def getBLOB(self, name):
            dev_name = getattr(self, "getDeviceName", lambda: None)()
            return MockPyIndi._properties.get(f"{dev_name}.{name}")
# Mock CONFIG for testing purposes
mock_config = {
    'development': {'dry_run': False},
    'hardware': {
        'indi_host': 'localhost',
        'indi_port': 7624,
        'mount_device_name': 'Mount',
        'camera_device_name': 'Camera',
        'focuser_device_name': 'Focuser',
        'max_slew_deg_s': 6.0
    },
    'observer': {
        'latitude_deg': 34.0,
        'longitude_deg': -118.0,
        'altitude_m': 100.0,
    },
    'camera_specs': {},
    'capture': {
        'autofocus': {
            'scan_range': 1,
            'step_base': 50,
            'sharpness_threshold': 1.0,
            'max_duration_s': 30.0,
            'exposure_s': 0.1,
        },
        'exposure_min_s': 0.001,
        'exposure_max_s': 10.0,
        'min_sequence_interval_s': 1.0,
        'num_sequence_images': 1,  # Simplified for testing
    },
    'selection': {
        'min_sun_separation_deg': 15.0,
        'min_elevation_deg': 5.0,
        'max_range_km': 100.0
    }
}

class TestIndiController(unittest.TestCase):
    """Exercises IndiController workflows (slew, autofocus, capture) with mocked devices."""

    current_time = time.time() # Class-level attribute for mock_time

    def setUp(self):
        # Reset mocked clock
        TestIndiController.current_time = time.time()

        # Clear mock storage to ensure test isolation
        MockPyIndi._properties = {}

        # Clear hardware_control and astro.coords from sys.modules to ensure fresh imports
        if 'hardware_control' in sys.modules:
            del sys.modules['hardware_control']
        if 'astro.coords' in sys.modules:
            del sys.modules['astro.coords']

        # Dynamically patch sys.modules for PyIndi and other dependencies
        self.sys_modules_patchers = [
            patch.dict('sys.modules', {'PyIndi': MockPyIndi}),
            patch.dict('sys.modules', {'cv2': MagicMock()}),
            patch.dict('sys.modules', {'numpy': MagicMock()}),
            patch.dict('sys.modules', {'astropy': MagicMock()}),
            patch.dict('sys.modules', {'astropy.units': MagicMock()}),
            patch.dict('sys.modules', {'astropy.coordinates': MagicMock()}),
            patch.dict('sys.modules', {'astropy.time': MagicMock()}),
            patch.dict('sys.modules', {'astropy.io': MagicMock()}),
            patch.dict('sys.modules', {'astropy.visualization': MagicMock()}),
            patch.dict('sys.modules', {'astropy.stats': MagicMock()}),
            patch.dict('sys.modules', {
                'geopy': MagicMock(),
                'geopy.distance': MagicMock(),
            }),
            patch.dict('sys.modules', {'imaging.analysis': MagicMock()}),
            patch.dict('sys.modules', {'imaging.stacking.orchestrator': MagicMock()}),
        ]
        for p in self.sys_modules_patchers:
            p.start()
            self.addCleanup(p.stop)

        # Seed essential properties so controller init does not timeout
        def seed_prop(dev_name, prop_name, prop):
            MockPyIndi._properties[f"{dev_name}.{prop_name}"] = prop

        # Camera BLOB property for detection
        blob_prop = MagicMock(name='CCD1')
        blob_prop.__len__.return_value = 0
        seed_prop('Camera', 'CCD1', blob_prop)

        # Camera upload mode switch (already ON)
        upload_prop = MagicMock()
        upload_prop.getDeviceName.return_value = 'Camera'
        upload_prop.name = 'UPLOAD_MODE'
        upload_prop.r = MockPyIndi.SR_1OFMANY
        upload_prop.size.return_value = 1
        upload_widget = MagicMock(s=MockPyIndi.ISS_ON)
        upload_prop.findWidgetByName.return_value = upload_widget
        upload_prop.__getitem__.return_value = upload_widget
        seed_prop('Camera', 'UPLOAD_MODE', upload_prop)

        # Mount coordinates baseline
        az_widget = MagicMock(value=0, min=0, max=360)
        alt_widget = MagicMock(value=0, min=-90, max=90)
        coord_prop = MagicMock(
            getDeviceName=lambda: 'Mount',
            name='HORIZONTAL_COORD',
            s=MockPyIndi.IPS_OK,
            findWidgetByName=lambda n: {'AZ': az_widget, 'ALT': alt_widget}.get(n)
        )
        seed_prop('Mount', 'HORIZONTAL_COORD', coord_prop)

        # Focuser position baseline
        focus_widget = MagicMock(value=1000, min=0, max=2000)
        focus_prop = MagicMock(
            getDeviceName=lambda: 'Focuser',
            name='ABS_FOCUS_POSITION',
            s=MockPyIndi.IPS_OK,
            findWidgetByName=lambda n: {'FOCUS_ABSOLUTE_POSITION': focus_widget}.get(n)
        )
        seed_prop('Focuser', 'ABS_FOCUS_POSITION', focus_prop)

        # Camera exposure baseline
        exp_widget_seed = MagicMock(value=0.1)
        exp_prop_seed = MagicMock(
            getDeviceName=lambda: 'Camera',
            name='CCD_EXPOSURE',
            s=MockPyIndi.IPS_IDLE,
            findWidgetByName=lambda n: {'CCD_EXPOSURE_VALUE': exp_widget_seed}.get(n)
        )
        seed_prop('Camera', 'CCD_EXPOSURE', exp_prop_seed)

        # Import IndiController *after* sys.modules are patched
        from hardware_control import IndiController
        self.IndiController = IndiController # Store for use in tests

        # Mock CONFIG globally for tests in this class
        self.config_patcher = patch('hardware_control.CONFIG', mock_config)
        self.mock_config = self.config_patcher.start()
        self.addCleanup(self.config_patcher.stop)

        # Mock functions imported into hardware_control.py
        self.patch_get_altaz_frame = patch('hardware_control.get_altaz_frame', MagicMock()).start()
        self.addCleanup(self.patch_get_altaz_frame.stop)
        self.patch_slew_time_needed = patch('hardware_control.slew_time_needed', MagicMock(return_value=10.0)).start()
        self.addCleanup(self.patch_slew_time_needed.stop)
        self.patch_get_sun_azel = patch('hardware_control.get_sun_azel', MagicMock(return_value=(0, 0))).start()
        self.addCleanup(self.patch_get_sun_azel.stop)
        self.patch_angular_sep_deg = patch('hardware_control.angular_sep_deg', MagicMock(return_value=100.0)).start()
        self.addCleanup(self.patch_angular_sep_deg.stop)
        self.patch_distance_km = patch('hardware_control.distance_km', MagicMock(return_value=10.0)).start()
        self.addCleanup(self.patch_distance_km.stop)
        self.patch_makedirs = patch('hardware_control.os.makedirs', MagicMock()).start()
        self.addCleanup(self.patch_makedirs.stop)
        self.patch_measure_sharpness = patch('hardware_control.measure_sharpness', MagicMock(return_value=50.0)).start()
        self.addCleanup(self.patch_measure_sharpness.stop)
        self.patch_save_png_preview = patch('hardware_control.save_png_preview', MagicMock()).start()
        self.addCleanup(self.patch_save_png_preview.stop)
        self.patch_schedule_stack_and_publish = patch('hardware_control.schedule_stack_and_publish', MagicMock()).start()
        self.addCleanup(self.patch_schedule_stack_and_publish.stop)
        self.patch_append_to_json = patch('hardware_control.append_to_json', MagicMock()).start()
        self.addCleanup(self.patch_append_to_json.stop)
        # Advance mocked time alongside sleeps inside hardware_control
        def fake_sleep(seconds):
            TestIndiController.current_time += seconds
            REAL_SLEEP(0)  # yield to other threads
        self.patch_sleep = patch('hardware_control.time.sleep', side_effect=fake_sleep)
        self.patch_sleep.start()
        self.addCleanup(self.patch_sleep.stop)

        # Instantiate the controller
        self.controller = self.IndiController()
        # Auto-advance property states for number commands to keep loops moving
        def auto_send_new_number(prop):
            dev = prop.getDeviceName()
            name = prop.name
            self._update_prop_state(dev, name, MockPyIndi.IPS_BUSY)
            self._update_prop_state(dev, name, MockPyIndi.IPS_OK)
        self.controller.sendNewNumber = auto_send_new_number

    def tearDown(self):
        pass # Cleanup handled by addCleanup

    def _create_mock_property(self, dev_name, prop_name, prop_type='number',
                               initial_state=MockPyIndi.IPS_IDLE, value=0):
        """Seed a fake INDI property (number/switch/blob) into the shared mock store."""
        mock_prop = MagicMock()
        mock_prop.getDeviceName.return_value = dev_name
        mock_prop.name = prop_name
        mock_prop.s = initial_state
        mock_prop.value = value  # For number properties
        mock_prop.findWidgetByName.return_value = MagicMock(value=value)  # For widgets

        if prop_type == 'switch':
            mock_prop.findWidgetByName.return_value = MagicMock(s=MockPyIndi.ISS_OFF)
            mock_prop.r = MockPyIndi.SR_1OFMANY  # Default rule
            mock_prop.size.return_value = 1  # For switch iteration
            mock_prop.__getitem__.return_value = MagicMock(s=MockPyIndi.ISS_OFF)
        elif prop_type == 'blob':
            mock_blob_item = MagicMock(bloblen=1, getblobdata=lambda: b'blob')
            mock_prop.__len__.return_value = 1
            mock_prop.__getitem__.return_value = mock_blob_item

        prop_key = f"{dev_name}.{prop_name}"
        MockPyIndi._properties[prop_key] = mock_prop
        return mock_prop

    def _update_prop_state(self, dev_name, prop_name, state):
        """Manually set property state and fire its event."""
        prop_key = f"{dev_name}.{prop_name}"
        with self.controller._property_states_lock:
            self.controller._property_states[prop_key] = state
        with self.controller._property_events_lock:
            if prop_key not in self.controller._property_events:
                self.controller._property_events[prop_key] = threading.Event()
            self.controller._property_events[prop_key].set()

    def test_wait_prop_event_driven(self):
        dev_name = 'Mount'
        prop_name = 'TEST_PROP'
        mock_dev = self.controller.getDevice(dev_name)

        # Property is not available initially
        initial_prop = self.controller.getNumber(dev_name, prop_name)
        self.assertIsNone(initial_prop)

        # Start a thread to wait for the property
        result = [None]

        def worker():
            result[0] = self.controller._wait_prop(
                MockPyIndi.BaseDevice.getNumber, mock_dev, prop_name, timeout=2)

        thread = threading.Thread(target=worker)
        thread.start()

        # Simulate delay
        time.sleep(0.1)
        self.assertIsNone(result[0])

        # Create and notify the property from a "different" context (like INDI server)
        mock_prop_val = self._create_mock_property(dev_name, prop_name, value=100)
        self.controller.newProperty(mock_prop_val)  # This should set the event

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(result[0])
        self.assertEqual(result[0].value, 100)

    def test_wait_prop_timeout(self):
        dev_name = 'Camera'
        prop_name = 'TIMEOUT_PROP'
        mock_dev = self.controller.getDevice(dev_name)

        result = [None]

        def worker():
            result[0] = self.controller._wait_prop(
                MockPyIndi.BaseDevice.getNumber, mock_dev, prop_name, timeout=0.5)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=1.0)  # Wait for thread to finish (should timeout)

        self.assertFalse(thread.is_alive())
        self.assertIsNone(result[0])

    @patch('hardware_control.time.time', side_effect=lambda: TestIndiController.current_time)
    def test_slew_to_az_el_non_blocking(self, mock_time):
        TestIndiController.current_time = time.time()  # Initialize mock time

        mount_dev_name = 'Mount'
        coord_prop_name = 'HORIZONTAL_COORD'
        az_widget = MagicMock(value=0, min=0, max=360)
        alt_widget = MagicMock(value=0, min=-90, max=90)
        mock_coord_prop = MagicMock(
            getDeviceName=lambda: mount_dev_name,
            name=coord_prop_name,
            s=MockPyIndi.IPS_IDLE,  # Initial state
            findWidgetByName=lambda name: {'AZ': az_widget, 'ALT': alt_widget}.get(name)
        )
        self._create_mock_property(mount_dev_name, coord_prop_name, initial_state=MockPyIndi.IPS_IDLE, value=0)
        MockPyIndi._properties[f"{mount_dev_name}.{coord_prop_name}"] = mock_coord_prop

        # Start slew in a separate thread
        slew_result = [False]

        def slew_worker():
            slew_result[0] = self.controller.slew_to_az_el(90, 45)

        thread = threading.Thread(target=slew_worker)
        thread.start()

        # Simulate time passing and mount becoming busy, then OK
        TestIndiController.current_time += 0.1
        self._update_prop_state(mount_dev_name, coord_prop_name, MockPyIndi.IPS_BUSY)

        # Slew should be in progress
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        self.assertFalse(slew_result[0])

        # Simulate slew completion
        TestIndiController.current_time += 1.0
        self._update_prop_state(mount_dev_name, coord_prop_name, MockPyIndi.IPS_OK)

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(slew_result[0])

    @patch('hardware_control.time.time', side_effect=lambda: TestIndiController.current_time)
    def test_autofocus_non_blocking_move(self, mock_time):
        TestIndiController.current_time = time.time()

        focuser_dev_name = 'Focuser'
        pos_prop_name = 'ABS_FOCUS_POSITION'
        pos_widget = MagicMock(value=1000, min=0, max=2000)
        mock_pos_prop = MagicMock(
            getDeviceName=lambda: focuser_dev_name,
            name=pos_prop_name,
            s=MockPyIndi.IPS_IDLE,
            findWidgetByName=lambda name: {'FOCUS_ABSOLUTE_POSITION': pos_widget}.get(name)
        )
        self._create_mock_property(focuser_dev_name, pos_prop_name, initial_state=MockPyIndi.IPS_IDLE, value=1000)
        MockPyIndi._properties[f"{focuser_dev_name}.{pos_prop_name}"] = mock_pos_prop

        # Mock capture_image to avoid actual image processing
        self.controller.capture_image = MagicMock(return_value='/fake/path/image.fits')
        # Mock measure_sharpness is already done in setUp
        autofocus_result = [False]

        def autofocus_worker():
            autofocus_result[0] = self.controller.autofocus()

        thread = threading.Thread(target=autofocus_worker)
        thread.start()

        # Simulate initial move to a scan position
        TestIndiController.current_time += 0.1
        self._update_prop_state(focuser_dev_name, pos_prop_name, MockPyIndi.IPS_BUSY)
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())

        # Simulate completion of first move
        TestIndiController.current_time += 0.5
        self._update_prop_state(focuser_dev_name, pos_prop_name, MockPyIndi.IPS_OK)

        # Allow time for next steps (capture, analyze, next move)
        time.sleep(1.0)

        # Simulate final move to best position (assumes best_pos is one of the scanned positions)
        TestIndiController.current_time += 0.1
        self._update_prop_state(focuser_dev_name, pos_prop_name, MockPyIndi.IPS_BUSY)
        time.sleep(0.1)
        TestIndiController.current_time += 0.5
        self._update_prop_state(focuser_dev_name, pos_prop_name, MockPyIndi.IPS_OK)

        thread.join(timeout=3.0)  # Longer timeout for multiple moves and captures
        self.assertFalse(thread.is_alive())
        self.assertTrue(autofocus_result[0])

    @patch('hardware_control.time.time', side_effect=lambda: TestIndiController.current_time)
    def test_capture_image_non_blocking(self, mock_time):
        TestIndiController.current_time = time.time()

        camera_dev_name = 'Camera'
        exp_prop_name = 'CCD_EXPOSURE'
        exp_widget = MagicMock(value=0.1)
        mock_exp_prop = MagicMock(
            getDeviceName=lambda: camera_dev_name,
            name=exp_prop_name,
            s=MockPyIndi.IPS_IDLE,
            findWidgetByName=lambda name: {'CCD_EXPOSURE_VALUE': exp_widget}.get(name)
        )
        self._create_mock_property(camera_dev_name, exp_prop_name, initial_state=MockPyIndi.IPS_IDLE, value=0.1)
        MockPyIndi._properties[f"{camera_dev_name}.{exp_prop_name}"] = mock_exp_prop

        # Mock BLOB property
        blob_prop_name = 'CCD1'
        mock_blob_prop = MagicMock(
            getDeviceName=lambda: camera_dev_name,
            name=blob_prop_name,
            s=MockPyIndi.IPS_IDLE,
            bloblen=1000,  # Simulate some blob data length
            getblobdata=lambda: b'mock_image_data'
        )
        mock_blob_item = MagicMock(bloblen=1000, getblobdata=lambda: b'mock_image_data')
        mock_blob_prop.__len__.return_value = 1
        mock_blob_prop.__getitem__.return_value = mock_blob_item
        self._create_mock_property(camera_dev_name, blob_prop_name, prop_type='blob', initial_state=MockPyIndi.IPS_IDLE)
        # Mocks for open, fits.open, and save_png_preview are already in setUp

        tmp_dir = tempfile.mkdtemp()
        capture_path = os.path.join(tmp_dir, "test_image.fits")
        capture_result = [None]

        def capture_worker():
            capture_result[0] = self.controller.capture_image(0.1, capture_path)

        thread = threading.Thread(target=capture_worker)
        thread.start()

        # Simulate exposure command being sent and camera becoming busy
        TestIndiController.current_time += 0.01
        self._update_prop_state(camera_dev_name, exp_prop_name, MockPyIndi.IPS_BUSY)
        time.sleep(0.01)
        self.assertTrue(thread.is_alive())

        # Simulate exposure command completing
        TestIndiController.current_time += 0.05
        self._update_prop_state(camera_dev_name, exp_prop_name, MockPyIndi.IPS_OK)

        # Simulate BLOB arriving after some delay
        time.sleep(0.05)  # Allow capture_image to proceed to blob wait
        self.assertTrue(thread.is_alive())

        # Wait until blob event is registered
        for _ in range(50):
            with self.controller._blob_events_lock:
                if len(self.controller._blob_events) > 0:
                    break
            REAL_SLEEP(0.01)

        TestIndiController.current_time += 0.1
        # Simulate the BLOB arrival by directly populating received data and firing the event
        with self.controller._exposure_queue.mutex:
            if self.controller._exposure_queue.queue:
                exposure_token = self.controller._exposure_queue.queue[0]
            else:
                self.fail("Exposure queue empty, capture_image did not register token.")
        with self.controller._blob_lock:
            self.controller._received_blobs[exposure_token] = b'mock_image_data'
        with self.controller._blob_events_lock:
            if exposure_token not in self.controller._blob_events:
                self.controller._blob_events[exposure_token] = threading.Event()
            self.controller._blob_events[exposure_token].set()

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(capture_result[0], capture_path)

if __name__ == '__main__':
    unittest.main()
