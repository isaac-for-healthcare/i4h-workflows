#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <solum/solum.h>

namespace py = pybind11;

// Global callback functions stored as std::functions
std::function<void(CusConnection, int, const char*)> connect_fn_cb;
std::function<void(int)> cert_fn_cb;
std::function<void(CusPowerDown, int)> power_down_fn_cb;
std::function<void(py::bytearray, int, int, int, double, double, double, double)> processed_image_fn_cb;
std::function<void(int)> imu_port_fn_cb;
std::function<void(const CusPosInfo*)> imu_data_fn_cb;
std::function<void(CusImagingState, int)> imaging_fn_cb;
std::function<void(CusButton, int)> buttons_fn_cb;
std::function<void(CusErrorCode, const char*)> error_fn_cb;

/**
 * @brief Wrapper function for handling connection callback.
 *
 * @param res Connection result.
 * @param port Port number.
 * @param status Connection status message.
 */
void connect_fn_wrapper(CusConnection res, int port, const char* status) {
    return connect_fn_cb(res, port, status);
}

/**
 * @brief Wrapper function for handling certificate callback.
 *
 * @param daysValid Number of days the certificate is valid.
 */
void cert_fn_wrapper(int daysValid) {
    return cert_fn_cb(daysValid);
}

/**
 * @brief Wrapper function for handling power down callback.
 *
 * @param res Power down reason.
 * @param tm Time in seconds until probe powers down, 0 for immediate shutdown.
 */
void power_down_fn_wrapper(CusPowerDown res, int tm) {
    return power_down_fn_cb(res, tm);
}

/**
 * @brief Wrapper function for handling processed image callback.
 *
 * @param img Pointer to the image data.
 * @param nfo Pointer to the processed image information.
 * @param npos Number of positions.
 * @param pos Pointer to position information.
 */
void processed_image_fn_wrapper(
        const void* img, const CusProcessedImageInfo* nfo, int npos, const CusPosInfo* pos) {
    const char* array = reinterpret_cast<const char*>(img);

    py::gil_scoped_acquire acquire;
    py::bytearray data(array, nfo->imageSize);
    return processed_image_fn_cb(
            data, nfo->width, nfo->height, nfo->imageSize,
            nfo->micronsPerPixel, nfo->originX, nfo->originY,
            nfo->fps);
}

/**
 * @brief Wrapper function for handling IMU port callback.
 *
 * @param port The new IMU data streaming port number.
 */
void imu_port_fn_wrapper(int port) {
    return imu_port_fn_cb(port);
}

/**
 * @brief Wrapper function for handling IMU data callback.
 *
 * @param pos The positional information data tagged with the image
 */
void imu_data_fn_wrapper(const CusPosInfo* pos) {
    return imu_data_fn_cb(pos);
}

/**
 * @brief Wrapper function for handling imaging state callback.
 *
 * @param state Current imaging state.
 * @param imaging 1 = running , 0 = stopped
 */
void imaging_fn_wrapper(CusImagingState state, int imaging) {
    return imaging_fn_cb(state, imaging);
}

/**
 * @brief Wrapper function for handling button callback.
 *
 * @param btn The button that was pressed.
 * @param clicks Number of clicks performed.
 */
void buttons_fn_wrapper(CusButton btn, int clicks) {
    return buttons_fn_cb(btn, clicks);
}

/**
 * @brief Wrapper function for handling error callback.
 *
 * @param code Error code.
 * @param msg Error message.
 */
void error_fn_wrapper(CusErrorCode code, const char* msg) {
    return error_fn_cb(code, msg);
}


/**
 * @brief A class to manage and invoke solum-related callbacks.
 */
struct Solum {

    /**
     * @brief Constructor to initialize Solum with callback functions.
     *
     * @param connect_fn Connection callback.
     * @param cert_fn Certificate callback.
     * @param power_down_fn Power down callback.
     * @param processed_image_fn Processed image callback.
     * @param imu_port_fn IMU port callback.
     * @param imu_data_fn IMU data callback.
     * @param imaging_fn Imaging state callback.
     * @param buttons_fn Button press callback.
     * @param error_fn Error callback.
     */
    Solum(std::function<void(CusConnection, int, const char*)> connect_fn,
          std::function<void(int)> cert_fn,
          std::function<void(CusPowerDown, int)> power_down_fn,
          std::function<void(py::bytearray, int, int, int, double, double, double, double)> processed_image_fn,
          std::function<void(int)> imu_port_fn,
          std::function<void(const CusPosInfo*)> imu_data_fn,
          std::function<void(CusImagingState, int)> imaging_fn,
          std::function<void(CusButton, int)> buttons_fn,
          std::function<void(CusErrorCode, const char*)> error_fn)
    {
        connect_fn_cb = connect_fn;
        cert_fn_cb = cert_fn;
        power_down_fn_cb = power_down_fn;
        processed_image_fn_cb = processed_image_fn;
        imu_port_fn_cb = imu_port_fn;
        imu_data_fn_cb = imu_data_fn;
        imaging_fn_cb = imaging_fn;
        buttons_fn_cb = buttons_fn;
        error_fn_cb = error_fn;
    }

    /**
     * @brief Destructor to clean up callback functions and release resources.
     */
    ~Solum() {
        connect_fn_cb = nullptr;
        cert_fn_cb = nullptr;
        power_down_fn_cb = nullptr;
        processed_image_fn_cb = nullptr;
        imu_port_fn_cb = nullptr;
        imu_data_fn_cb = nullptr;
        imaging_fn_cb = nullptr;
        buttons_fn_cb = nullptr;
        error_fn_cb = nullptr;
        solumDestroy();
    }

    /**
     * @brief Initializes Solum with the given parameters.
     *
     * @param path Path to the store directory.
     * @param width Width of the image.
     * @param height Height of the image.
     * @return true if initialization succeeds, false otherwise.
     */
    bool init(std::string path, int width, int height) {
        std::cout << "Clarius: Initializing Caster with path: " << path
            << ", width: " << width
            << ", height: " << height << std::endl;

        auto initParams = solumDefaultInitParams();
        initParams.args.argc = 0;
        initParams.args.argv = nullptr;
        initParams.storeDir = strdup(path.c_str());
        initParams.connectFn = connect_fn_wrapper;
        initParams.certFn = cert_fn_wrapper;
        initParams.powerDownFn = power_down_fn_wrapper;
        initParams.newProcessedImageFn = processed_image_fn_wrapper;
        initParams.newRawImageFn = nullptr;
        initParams.newImuPortFn = imu_port_fn_wrapper;
        initParams.newImuDataFn = imu_data_fn_wrapper;
        initParams.imagingFn = imaging_fn_wrapper;
        initParams.buttonFn = buttons_fn_wrapper;
        initParams.errorFn = error_fn_wrapper;
        initParams.width = width;
        initParams.height = height;
        // initialize with callbacks
        if (solumInit(&initParams) < 0)
        {
            std::cout << "Clarius: Could not initialize solum module" << std::endl;
            return false;
        }
        if (solumSetFormat(Jpeg) < 0)
        {
            std::cout << "Clarius: Could not set format to JPEG" << std::endl;
            return false;
        }
        return true;  // Assuming initialization succeeds
    }

    /**
     * @brief Establishes a connection to the probe.
     *
     * @param ip IP address of the probe.
     * @param port Port number.
     * @param mode Mode of connection.
     * @return true if connection request is made, false otherwise.
     */
    bool connect(std::string ip, int port, std::string mode) {
        std::cout << "Clarius: Connecting to " << ip << ":" << port << " in mode " << mode << std::endl;
        auto connectParams = solumDefaultConnectionParams();
        connectParams.ipAddress = ip.c_str();
        connectParams.port = port;
        if (solumConnect(&connectParams) < 0) {
            std::cout << "Clarius: Error calling connect" << std::endl;
        } else {
            std::cout << "Clarius: Trying to connect..." << std::endl;
        }

        return true;
    }


    /**
     * @brief Disconnects from the probe.
     */
    void disconnect() {
        solumDisconnect();
    }

    /**
     * @brief Loads the specified application on the probe.
     *
     * @param probe_model Model of the probe.
     * @param application Application name.
     * @return true if the application is loaded successfully, false otherwise.
     */
    bool load_application(const std::string& probe_model, const std::string& application) {

        std::cout << "Clarius: Model: " << probe_model << " Application: " << application << std::endl;
        if (solumLoadApplication(probe_model.c_str(), application.c_str()) == 0) {
            std::cout << "Clarius: Trying to load application '" << application << "'" << std::endl;
            return true;
        } else {
            std::cout << "Clarius: Error calling load application" << std::endl;
            return false;
        }
    }

    /**
     * @brief Sets the certificate for the probe.
     *
     * @param cert Path to the certificate file.
     * @return true if certificate is set successfully, false otherwise.
     */
    bool set_certificate(std::string cert) {
        std::ifstream fs(cert);
        if (!fs.is_open()) {
            std::cout << "Clarius: Error loading certificate file: " << cert << std::endl;
            return false;
        } else {
            std::stringstream ss;
            ss << fs.rdbuf();
            if (solumSetCert(ss.str().c_str()) < 0) {
                std::cout << "Clarius: Error sending certificate" << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Starts imaging on the probe.
     *
     * @return true if imaging starts successfully, false otherwise.
     */
    bool run_imaging() {
        if (solumRun(1) < 0) {
            std::cout << "Clarius: Run request failed." << std::endl;
            return false;
        } else {
            std::cout << "Clarius: Run imaging..." << std::endl;
            return true;
        }
    }

    /**
     * @brief Stops imaging on the probe.
     *
     * @return true if imaging stops successfully, false otherwise.
     */
    bool stop_imaging() {
        if (solumRun(0) < 0) {
            std::cout << "Clarius: Stop request failed." << std::endl;
            return false;
        } else {
            std::cout << "Clarius: Stop imaging..." << std::endl;
            return true;
        }
    }
};

PYBIND11_MODULE(pysolum, m) {
    // Enum bindings
    py::enum_<CusConnection>(m, "CusConnection", "Represents the connection status of the probe.")
        .value("ConnectionError", ConnectionError, "Error connecting to the probe.")
        .value("ProbeConnected", ProbeConnected, "The probe is connected successfully.")
        .value("ProbeDisconnected", ProbeDisconnected, "The probe has been disconnected.")
        .value("ConnectionFailed", ConnectionFailed, "The connection attempt has failed.")
        .value("SwUpdateRequired", SwUpdateRequired, "Software update is required.")
        .value("OSUpdateRequired", OSUpdateRequired, "Operating system update is required.")
        .export_values();

    py::enum_<CusPowerDown>(m, "CusPowerDown", "Represents reasons for the probe's power down state.")
        .value("Idle", Idle, "Indicates that the probe is in idle state.")
        .value("TooHot", TooHot, "Indicates that the probe is shutting down due to high temperature.")
        .value("LowBattery", LowBattery, "Indicates that the probe is shutting down due to low battery.")
        .value("ButtonOff", ButtonOff, "Indicates that the probe is shutting down due to button press.")
        .value("ChargingInDock", ChargingInDock, "Indicates that probe was docked in charger.")
        .value("SoftwareShutdown", SoftwareShutdown, "Indicates that the probe was shut down via software command.")
        .export_values();

    py::enum_<CusButton>(m, "CusButton", "Represents the probe buttons.")
        .value("ButtonUp", ButtonUp, "The Up Button.")
        .value("ButtonDown", ButtonDown, "The Down Button.")
        .value("ButtonHandle", ButtonHandle, "Handle Button (custom probes only).")
        .export_values();

    py::enum_<CusImagingState>(m, "CusImagingState", "Represents the different states of imaging on the probe.")
        .value("ImagingNotReady", ImagingNotReady, "Imaging is not ready, probe and application need to be loaded.")
        .value("ImagingReady", ImagingReady, "Imaging is ready.")
        .value("CertExpired", CertExpired, "Cannot image due to expired certificate.")
        .value("PoorWifi", PoorWifi, "Stopped imaging due to poor Wi-Fi.")
        .value("NoContact", NoContact, "Stopped imaging due to no patient contact detected.")
        .value("ChargingChanged", ChargingChanged, "Probe started running or stopped due to change in charging status.")
        .value("LowBandwidth", LowBandwidth, "Low bandwidth was detected, imaging parameters were adjusted.")
        .value("MotionSensor", MotionSensor, "Probe started running or stopped due to change in motion sensor.")
        .value("NoTee", NoTee, "Cannot image due to tee being disconnected.")
        .value("TeeExpired", TeeExpired, "Cannot image due to tee being expired.")
        .export_values();

    py::enum_<CusErrorCode>(m, "CusErrorCode", "Represents various error codes that may occur during operation.")
        .value("ErrorGeneric", ErrorGeneric, "Generic error.")
        .value("ErrorSetup", ErrorSetup, "Setup error.")
        .value("ErrorProbe", ErrorProbe, "Probe error.")
        .value("ErrorApplication", ErrorApplication, "Application load error.")
        .value("ErrorSwUpdate", ErrorSwUpdate, "Software update error.")
        .value("ErrorGl", ErrorGl, "GL error.")
        .value("ErrorRawData", ErrorRawData, "Raw data error.")
        .export_values();

    // Class bindings
    py::class_<Solum>(m, "Solum")
        .def(py::init<
                std::function<void(CusConnection, int, const char*)>,
                std::function<void(int)>,
                std::function<void(CusPowerDown, int)>,
                std::function<void(py::bytearray, int, int, int, double, double, double, double)>,
                std::function<void(int)>,
                std::function<void(const CusPosInfo*)>,
                std::function<void(CusImagingState, int)>,
                std::function<void(CusButton, int)>,
                std::function<void(CusErrorCode, const char*)>
                >(),
                py::arg("connect_cb"),
                py::arg("cert_cb"),
                py::arg("power_down_cb"),
                py::arg("processed_image_cb"),
                py::arg("imu_port_cb"),
                py::arg("imu_data_cb"),
                py::arg("imaging_cb"),
                py::arg("button_cb"),
                py::arg("error_cb"))
        .def("init", &Solum::init)
        .def("connect", &Solum::connect)
        .def("disconnect", &Solum::disconnect)
        .def("load_application", &Solum::load_application)
        .def("set_certificate", &Solum::set_certificate)
        .def("run_imaging", &Solum::run_imaging)
        .def("stop_imaging", &Solum::stop_imaging);
}
