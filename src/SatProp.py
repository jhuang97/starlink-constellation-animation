import os
import sys
import time
from ctypes import *
from AstroUtils import *
from wrappers.DllMainWrapper import *
from wrappers.EnvConstWrapper import *
from wrappers.AstroFuncWrapper import *
from wrappers.TimeFuncWrapper import *
from wrappers.TleWrapper import *
from wrappers.Sgp4PropWrapper import *
import numpy as np
from my_util import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import math


class Sgp4Prop(object):
    # Constants used when we need to pass an integer control value to a function
    DEFAULT_STRING_LENGTH = 513  # 512 + 1 for NULL
    ELSET_LOAD_ORDER = 2
    PROPOUT_NODAL_AP_PER = 2
    PROPOUT_MEAN_ELEM = 3
    PROPOUT_OSC_ELEM = 4

    # Global constants specifying the output file names
    OSC_STATE_FILE = 'OscState.txt'
    OSC_ELEM_FILE = 'OscElem.txt'
    MEAN_ELEM_FILE = 'MeanElem.txt'
    LLH_ELEM_FILE = 'LatLonHeight.txt'
    NODAL_AP_PER_FILE = 'NodalApPer.txt'

    # the name of the log file to open
    LOG_FILE_NAME = 'sgp4prop.log'

    # time tolerance
    EPSI = 0.00500

    # Global variables used to store the dll objects
    # We place them here so that all of our functions will have access to them, although they're initially set to None
    DllMain = None
    EnvConst = None
    AstroFunc = None
    TimeFunc = None
    Tle = None
    Sgp4Prop = None

    # These make it easier for routines that write to the output files
    # Since there are 5 of them, we chose to define these globally instead of
    # passing 5 parameters to all of the routines that need them.
    # We define 5 file objects, as well as a list that will contain them to make it easier for routines to write
    # the same data to all 5 files since this is a common operation.
    oscStateFile = None
    oscElemFile = None
    meanElemFile = None
    llhElemFile = None
    nodalApPerFile = None
    outfiles = {}

    def __init__(self, mode, input_file_name, n_day_divs=32, write_text_files=False,
                 end_dt=None, backtrack=False, save_tle_epochs=False):
        self.mode = mode
        self.input_file_name = input_file_name
        self.write_text_files = write_text_files
        self.n_day_divs = n_day_divs
        self.end_dt = end_dt
        self.backtrack = backtrack  # also propagates each TLE backwards and linearly interpolates
        self.save_tle_epochs = save_tle_epochs  # saves time index and epochs of the TLEs used

    def run(self):
        # Print startup message and process command line
        # print('Program starts.')

        # time.clock() was deprecated in 3.3
        if sys.version_info[0] > 3 or (sys.version_info[0] == 3 and sys.version_info[1] >= 3):
            start = time.process_time()
        else:
            start = time.clock()

        inputFileName = self.input_file_name
        outputDir = '../output'
        # Path to dll's
        libPath = '/[directory]/Sgp4Prop_small_v8.0/Sgp4Prop_small/Lib/Linux64'
        # also need to set the LD_LIBRARY_PATH environmental variable to the value of libPath - this is done when
        # you initially run Python

        # Print the parameters
        print('Input File=%s' % inputFileName)
        # print('Output Directory=%s' % outputDir)
        # print('Library Path=%s\n' % libPath)

        dlclose_func = cdll.LoadLibrary('').dlclose
        dlclose_func.argtypes = [c_void_p]
        # Load and initialize dll's
        self.DllMain = LoadDllMainDll(libPath)
        self.EnvConst = LoadEnvConstDll(libPath)
        self.TimeFunc = LoadTimeFuncDll(libPath)
        self.AstroFunc = LoadAstroFuncDll(libPath)
        self.Tle = LoadTleDll(libPath)
        self.Sgp4Prop = LoadSgp4PropDll(libPath)
        handles = [self.DllMain._handle, self.EnvConst._handle, self.TimeFunc._handle,
                   self.AstroFunc._handle, self.Tle._handle, self.Sgp4Prop._handle]
        apPtr = self.DllMain.DllMainInit()
        for initFunction in [self.EnvConst.EnvInit, self.TimeFunc.TimeFuncInit, self.AstroFunc.AstroFuncInit,
                             self.Tle.TleInit, self.Sgp4Prop.Sgp4Init]:
            if initFunction(apPtr):
                ShowMsgAndTerminate(self.DllMain)

        # Open the log file
        if self.DllMain.OpenLogFile(self.LOG_FILE_NAME.encode("UTF-8")):
            ShowMsgAndTerminate(self.DllMain)

        # Load TLE's and the 6P-Card from the input file
        if self.Tle.TleLoadFile(inputFileName.encode("UTF-8")):
            ShowMsgAndTerminate(self.DllMain)
        if self.TimeFunc.TConLoadFile(inputFileName.encode("UTF-8")):
            ShowMsgAndTerminate(self.DllMain)

        # Load the satellite keys we loaded into a ctypes array for later processing
        # numSats = c_int()
        numSats = self.Tle.TleGetCount()
        satKeys = CreateCArray(c_longlong, [numSats])
        # print(numSats, 'num keys', len(satKeys))
        self.Tle.TleGetLoaded(self.ELSET_LOAD_ORDER, satKeys)

        # Retrieve information about Sgp4Prop.dll
        # We'll print it here, and also use it in file headers
        cInfo = create_string_buffer(self.DEFAULT_STRING_LENGTH)
        self.Sgp4Prop.Sgp4GetInfo(cInfo)
        info = cInfo.value.rstrip()
        # print(info)

        if self.write_text_files:
            # Open output files and print header
            self.oscStateFile = self.OpenOutputFile(os.path.join(outputDir, self.OSC_STATE_FILE))
            self.oscElemFile = self.OpenOutputFile(os.path.join(outputDir, self.OSC_ELEM_FILE))
            self.meanElemFile = self.OpenOutputFile(os.path.join(outputDir, self.MEAN_ELEM_FILE))
            self.llhElemFile = self.OpenOutputFile(os.path.join(outputDir, self.LLH_ELEM_FILE))
            self.nodalApPerFile = self.OpenOutputFile(os.path.join(outputDir, self.NODAL_AP_PER_FILE))

            self.outFiles = [self.oscStateFile, self.oscElemFile, self.meanElemFile, self.llhElemFile, self.nodalApPerFile]
            self.PrintHeaders(info, inputFileName)

        # Process each TLE
        key_times, time_indices = self.plan_times(satKeys, self.n_day_divs)
        if self.mode == 'reference_sat':
            shift_end_time = days_since_yr(self.end_dt, 1950)
            time_indices = np.append(time_indices, round((shift_end_time - key_times[0]) * self.n_day_divs))
        if self.save_tle_epochs:
            osc_elems, mean_elems, tle_tidxs, tle_epochs = self.prop_satellite_tles(satKeys, self.n_day_divs, key_times, time_indices)
        else:
            osc_elems, mean_elems = self.prop_satellite_tles(satKeys, self.n_day_divs, key_times, time_indices)
        if self.mode == 'reference_sat':
            key_times = np.append(key_times, shift_end_time)
        times_ds50 = np.linspace(key_times[0], key_times[-1], time_indices[-1]+1)
        times_dt = [ds50UTC_to_datetime(dt) for dt in times_ds50]

        # for key in satKeys:
        #     self.PropagateSatellite(key)

        # Clean up and print exit message
        self.Sgp4Prop.Sgp4RemoveAllSats()
        if self.write_text_files:
            for outFile in self.outFiles:
                outFile.close()
        self.DllMain.CloseLogFile()
        # print('Program completed successfully.')

        # time.clock() was deprecated in 3.3
        if sys.version_info[0] > 3 or (sys.version_info[0] == 3 and sys.version_info[1] >= 3):
            finish = time.process_time()
        else:
            finish = time.clock()

        print('Total run time = %10.2f seconds' % (finish - start))
        if self.mode == 'test_figures':
            # plt.plot(times_ds50, osc_elems[:, 0], '.')
            long_past_asc_node = np.remainder(osc_elems[:, 4] + osc_elems[:, 5], 360.)
            # plt.plot(times_ds50, long_past_asc_node)
            # plt.xlim(25521, 25522)

            plt.plot(times_dt, mean_elems[:, 0], '.-')
            # plt.plot(times_dt, osc_elems[:, 4], '.-')
            # plt.plot(times_dt, long_past_asc_node, '.-')
            ax = plt.gca()
            # plt.xlim(datetime(2020,1,1), datetime(2020,1,2))
            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.xlabel('time')
            # plt.ylabel('semimajor axis (km)')
            plt.show()

        # unload all the DLLs
        del self.DllMain
        del self.EnvConst
        del self.TimeFunc
        del self.AstroFunc
        del self.Tle
        del self.Sgp4Prop
        for h in handles:
            dlclose_func(h)

        if self.save_tle_epochs:
            return times_dt, mean_elems, osc_elems, tle_tidxs, tle_epochs
        else:
            return times_dt, mean_elems, osc_elems

    def OpenOutputFile(self, fileName):
        """Opens the specified file for writing.  If the file can not be opened, the function terminates the program
        with a status of 1. If the file already exists, it will be over written.
        """
        try:
            fileObj = open(fileName, 'w')
        except IOError:
            ShowMsgAndTerminate('Unable to open output file ' + fileName)
        return fileObj

    def PrintHeaders(self, info, inputFileName):
        """Prints the header at the top of each output file.

        This function handles both the standard and file-specific headers.

        Parameters
        info -- A string containing the info about Sgp4Prop.dll.
        inputFileName -- A string containing the name of the input file.

        """
        # Load the individual elements of the 6P card
        pStartFrEpoch = c_int()
        pStopFrEpoch = c_int()
        pStartTime = c_double()
        pStopTime = c_double()
        pStepSize = c_double()
        self.TimeFunc.Get6P(byref(pStartFrEpoch), byref(pStopFrEpoch), byref(pStartTime), byref(pStopTime),
                            byref(pStepSize))
        # Write the common portion of the header to the output files
        for outFile in self.outFiles:
            PrintWarning('SGP4', outFile)
            outFile.write('%s\n\n\n' % info)
            outFile.write('EPHEMERIS GENERATED BY SGP4 USING THE WGS-72 EARTH MODEL\n')
            outFile.write('COORDINATE FRAME=TRUE EQUATOR AND MEAN EQUINOX OF EPOCH\n')
            outFile.write('USING THE FK5 MEAN OF J2000 TIME AND REFERENCE FRAME\n\n\n')
            outFile.write('INPUT FILE = %s\n' % inputFileName)
            if pStartFrEpoch:
                outFile.write('Start time = %14.4f min from epoch\n' % pStartTime.value)
            else:
                pStartDTG20Str = create_string_buffer(self.DEFAULT_STRING_LENGTH)
                self.TimeFunc.UTCToDTG20(pStartTime, pStartDTG20Str)
                outFile.write('Start time = %s\n' % pStartDTG20Str.value.rstrip())
            if (pStopFrEpoch):
                outFile.write('Stop time = %14.4f min from epoch\n' % pStopTime.value)
            else:
                pStopDTG20Str = create_string_buffer(self.DEFAULT_STRING_LENGTH)
                self.TimeFunc.UTCToDTG20(pStopTime, pStopDTG20Str)
                outFile.write('Stop time = %s\n' % pStopDTG20Str.value.rstrip())
            outFile.write('Step size = %14.4f min\n\n' % pStepSize.value)

        # Write the file-specific portion of the header to each output file
        self.oscStateFile.write(
            '     TSINCE (MIN)           X (KM)           Y (KM)           Z (KM)      XDOT (KM/S)       YDOT(KM/S)    ZDOT (KM/SEC)\n')
        self.oscElemFile.write(
            '     TSINCE (MIN)           A (KM)          ECC (-)        INC (DEG)       NODE (DEG)      OMEGA (DEG)   TRUE ANOM(DEG)\n')
        self.meanElemFile.write(
            '     TSINCE (MIN)     N (REVS/DAY)          ECC (-)        INC (DEG)       NODE (DEG)      OMEGA (DEG)         MA (DEG)\n')
        self.llhElemFile.write(
            '     TSINCE (MIN)         LAT(DEG)        LON (DEG)          HT (KM)           X (KM)           Y (KM)           Z (KM)\n')
        self.nodalApPerFile.write(
            '     TSINCE (MIN)   NODAL PER(MIN)1/NODAL(REVS/DAY)       N(REVS/DY)    ANOM PER(MIN)      APOGEE (KM)      PERIGEE(KM)\n\n')

    def plan_times(self, keys, n_day_divs):
        start_times = []
        for key in keys:
            # Initialize
            if self.Sgp4Prop.Sgp4InitSat(key):
                ShowMsgAndTerminate(self.DllMain)
            # Calculate start/stop time and step size based on this TLE and the 6P card
            pStartFrEpoch = c_int()
            pStopFrEpoch = c_int()
            pStartTime = c_double()
            pStopTime = c_double()
            pStepSize = c_double()
            self.TimeFunc.Get6P(byref(pStartFrEpoch), byref(pStopFrEpoch), byref(pStartTime), byref(pStopTime),
                                byref(pStepSize))
            epochDs50UTCStr = create_string_buffer(self.DEFAULT_STRING_LENGTH)
            self.Tle.TleGetField(key, 4, epochDs50UTCStr)
            epochDs50UTC = self.TimeFunc.DTGToUTC(epochDs50UTCStr)

            if pStartFrEpoch.value:
                startTime = epochDs50UTC + (pStartTime.value / 1440)
            else:
                startTime = pStartTime.value

            if self.Sgp4Prop.Sgp4RemoveSat(key):
                ShowMsgAndTerminate(self.DllMain)

            start_times.append(startTime)
        # print(start_times)
        # time_arr = np.array(start_times)
        # print(time_arr[1:] - time_arr[:-1])
        shifted_times = np.array([shift_time(ds, n_day_divs) for ds in start_times])
        time_indices = np.rint((shifted_times - shifted_times[0]) * n_day_divs).astype(int)
        return shifted_times, time_indices

    def prop_satellite_tles(self, keys, n_day_divs, key_times, time_indices):
        n_ticks = time_indices[-1]+1
        # if self.mode == 'reference_sat':
        #     n_ticks = time_indices[-1]
        # true_anomalies = np.zeros(n_ticks)
        print('# ticks:', n_ticks, '  # TLEs:', len(keys), len(time_indices))
        # print(time_indices)
        osc_elems = np.zeros((n_ticks, 6))
        mean_elems = np.zeros((n_ticks, 6))
        if self.backtrack:
            osc_elems_back = np.zeros((n_ticks, 6))
            mean_elems_back = np.zeros((n_ticks, 6))
            backtrack_weight = np.zeros(n_ticks)

        prev_tidxs = None
        # Set up ctypes variables we'll need for the propagation
        ds50UTC = c_double()
        mse = c_double()
        pos = CreateCArray(c_double, [3])
        vel = CreateCArray(c_double, [3])
        llh = CreateCArray(c_double, [3])  # latitude, longitude, height
        oscElem = CreateCArray(c_double, [6])
        meanElem = CreateCArray(c_double, [6])
        nodalApPer = CreateCArray(c_double, [3])

        oscElem_np = np.ctypeslib.as_array(oscElem)
        meanElem_np = np.ctypeslib.as_array(meanElem)

        if self.save_tle_epochs:
            tle_tidxs = []
            tle_epochs = []

        def propagate_ticks(tidxs, osc_elems_out, mean_elems_out):
            tidx = None
            for tidx in tidxs:
                ds50UTC.value = key_times[0] + tidx / n_day_divs
                # Propagate the satellite
                if self.Sgp4Prop.Sgp4PropDs50UTC(c_longlong(key), ds50UTC, byref(mse), pos, vel, llh):
                    # Decay condition, print error and move to next satellite
                    propErrMsg = create_string_buffer(self.DEFAULT_STRING_LENGTH)
                    self.DllMain.GetLastErrMsg(propErrMsg)
                    print(propErrMsg.value.rstrip())
                    return tidx-1, True  # Stop propagating

                # Compute/retrieve other propagator output data
                self.Sgp4Prop.Sgp4GetPropOut(key, self.PROPOUT_OSC_ELEM, oscElem)
                self.Sgp4Prop.Sgp4GetPropOut(key, self.PROPOUT_MEAN_ELEM, meanElem)
                self.Sgp4Prop.Sgp4GetPropOut(key, self.PROPOUT_NODAL_AP_PER, nodalApPer)
                trueAnomaly = self.AstroFunc.CompTrueAnomaly(oscElem)
                # true_anomalies[tidx] = trueAnomaly
                meanMotion = self.AstroFunc.AToN(c_double(meanElem[0]))
                # print(tidx)
                osc_elems_out[tidx, :5] = oscElem_np[[0, 1, 2, 4, 5]]
                osc_elems_out[tidx, 5] = trueAnomaly
                mean_elems_out[tidx, 0] = meanMotion
                mean_elems_out[tidx, 1:] = meanElem_np[[1, 2, 4, 5, 3]]

                # Check if height is below 100 km, stop propagation if so
                if llh[2] < 100.0:
                    # Print an error
                    if llh[2] < 0.0:
                        print("Warning: Decay condition. Distance from the Geoid (Km) = %10.3f" % llh[2])
                    else:
                        print("Warning: Height is low. HT (Km) = %10.3f" % llh[2])
                    return tidx, True  # Stop propagating
            return tidx, False

        for idx, key in enumerate(keys):
            skip = False
            if self.mode == 'reference_sat':
                tidxs = range(time_indices[idx], time_indices[idx + 1]+1)  # include end time
            elif idx < len(time_indices) - 1:
                if time_indices[idx] >= time_indices[idx+1]:
                    skip = True
                    tidxs = []
                else:
                    tidxs = range(time_indices[idx], time_indices[idx + 1])
            elif idx == len(time_indices)-1:
                tidxs = [n_ticks - 1]
            if len(tidxs) == 0:
                skip = True

            if not skip:
                # just in case
                line1 = create_string_buffer(self.DEFAULT_STRING_LENGTH)
                line2 = create_string_buffer(self.DEFAULT_STRING_LENGTH)
                self.Tle.TleGetLines(c_longlong(key), line1, line2)

                # Initialize
                if self.Sgp4Prop.Sgp4InitSat(key):
                    ShowMsgAndTerminate(self.DllMain)

                end_tick, did_decay = propagate_ticks(tidxs, osc_elems, mean_elems)
                if did_decay:
                    tidxs = range(tidxs[0], end_tick+1)
                if self.backtrack and prev_tidxs is not None:
                    assert len(tidxs) > 0
                    t_curr = tidxs[0]
                    t_prev = prev_tidxs[0]
                    if t_prev < 0 or t_curr <= t_prev:
                        print(tidxs[0])
                        print(line1[18:32])
                        print(prev_tidxs)
                    assert t_prev >= 0
                    assert t_curr > t_prev
                    n_backticks = t_curr - t_prev
                    tidx_back = range(t_prev+1, t_curr)
                    backtrack_weight[t_prev+1:t_curr] = np.arange(1, n_backticks)/n_backticks
                    end_tick, did_decay = propagate_ticks(tidx_back, osc_elems_back, mean_elems_back)
                if self.save_tle_epochs:
                    tle_tidxs.append(tidxs[0])
                    tle_epochs.append(line1[18:32])

                prev_tidxs = tidxs

                # Remove the satellite from the propagator since we're finished with it
                if self.Sgp4Prop.Sgp4RemoveSat(key):
                    ShowMsgAndTerminate(self.DllMain)

        if self.backtrack:  # linearly interpolate between successive TLEs
            b_weight = backtrack_weight[:, np.newaxis]
            osc_elems[:, :3] = osc_elems[:, :3] * (1-b_weight) + osc_elems_back[:, :3] * b_weight
            mean_elems[:, :3] = mean_elems[:, :3] * (1 - b_weight) + mean_elems_back[:, :3] * b_weight
            osc_elems[:, 3:] = circular_interp(osc_elems[:, 3:], osc_elems_back[:, 3:], b_weight)
            mean_elems[:, 3:] = circular_interp(mean_elems[:, 3:], mean_elems_back[:, 3:], b_weight)

        if self.save_tle_epochs:
            return osc_elems, mean_elems, np.array(tle_tidxs), np.array(tle_epochs)
        else:
            return osc_elems, mean_elems

    def PropagateSatellite(self, key):
        """Propagates a satellite to the times specified in the TLE.

        The results will be written to the 5 output files.

        Parameters
        key -- The satellite key. this must exist in the currently loaded set of TLE's. It will be wrapped in
        a c_longlong before it is looked up because of the fact it was converted to a standard Python integer
        by the TLE iteration loop in run().

        """
        # Print the TLE to all output files
        line1 = create_string_buffer(self.DEFAULT_STRING_LENGTH)
        line2 = create_string_buffer(self.DEFAULT_STRING_LENGTH)
        self.Tle.TleGetLines(c_longlong(key), line1, line2)
        if self.write_text_files:
            for outFile in self.outFiles:
                outFile.write('\n\n %s\n %s\n' % (line1.value.rstrip(), line2.value.rstrip()))

        # Initialize
        if self.Sgp4Prop.Sgp4InitSat(key):
            ShowMsgAndTerminate(self.DllMain)

        # Calculate start/stop time and step size based on this TLE and the 6P card
        pStartFrEpoch = c_int()
        pStopFrEpoch = c_int()
        pStartTime = c_double()
        pStopTime = c_double()
        pStepSize = c_double()
        self.TimeFunc.Get6P(byref(pStartFrEpoch), byref(pStopFrEpoch), byref(pStartTime), byref(pStopTime),
                            byref(pStepSize))
        epochDs50UTCStr = create_string_buffer(self.DEFAULT_STRING_LENGTH)
        self.Tle.TleGetField(key, 4, epochDs50UTCStr)
        epochDs50UTC = self.TimeFunc.DTGToUTC(epochDs50UTCStr)

        print(epochDs50UTCStr.value)
        print(epochDs50UTC)
        this_dt = ds50UTC_to_datetime(epochDs50UTC)
        print(this_dt, days_since_yr(this_dt, 2019))

        if pStartFrEpoch.value:
            startTime = epochDs50UTC + (pStartTime.value / 1440)
        else:
            startTime = pStartTime.value
        if pStopFrEpoch.value:
            stopTime = epochDs50UTC + (pStopTime.value / 1440)
        else:
            stopTime = pStopTime.value
        if startTime > stopTime:
            stepSize = -abs(pStepSize.value)
        else:
            stepSize = abs(pStepSize.value)

        # Set up ctypes variables we'll need for the propagation
        ds50UTC = c_double()
        mse = c_double()
        pos = CreateCArray(c_double, [3])
        vel = CreateCArray(c_double, [3])
        llh = CreateCArray(c_double, [3])  # latitude, longitude, height
        oscElem = CreateCArray(c_double, [6])
        meanElem = CreateCArray(c_double, [6])
        nodalApPer = CreateCArray(c_double, [3])

        # create numpy array that shares the same memory
        # pos_np = np.ctypeslib.as_array(pos)
        oscElem_np = np.ctypeslib.as_array(oscElem)

        # Propagate the satellite over the requested steps
        step = 0
        ds50UTC.value = startTime
        print('start time: ', startTime, ', stop time: ', stopTime, ', step: ', stepSize)
        while (stepSize >= 0 and ds50UTC.value < stopTime) or (stepSize < 0 and ds50UTC.value > stopTime):
            # Compute time to which to propagate, adjusting for tolerance
            ds50UTC.value = startTime + (step * stepSize / 1440.0)
            if (stepSize >= 0 and ds50UTC.value + (self.EPSI / 86400) > stopTime) or (
                    stepSize < 0 and ds50UTC.value - (self.EPSI / 86400) < stopTime):
                ds50UTC.value = stopTime
            # print(ds50UTC.value)

            # Propagate the satellite
            if self.Sgp4Prop.Sgp4PropDs50UTC(c_longlong(key), ds50UTC, byref(mse), pos, vel, llh):
                # Decay condition, print error and move to next satellite
                propErrMsg = create_string_buffer(self.DEFAULT_STRING_LENGTH)
                self.DllMain.GetLastErrMsg(propErrMsg)
                print(propErrMsg.value.rstrip())
                if self.write_text_files:
                    for outFile in self.outFiles:
                        outFile.write(str(propErrMsg.value.rstrip()) + "\n")
                break  # Stop propagating

            # print(pos_np)

            # Compute/retrieve other propagator output data
            self.Sgp4Prop.Sgp4GetPropOut(key, self.PROPOUT_OSC_ELEM, oscElem)
            self.Sgp4Prop.Sgp4GetPropOut(key, self.PROPOUT_MEAN_ELEM, meanElem)
            self.Sgp4Prop.Sgp4GetPropOut(key, self.PROPOUT_NODAL_AP_PER, nodalApPer)
            trueAnomaly = self.AstroFunc.CompTrueAnomaly(oscElem)
            meanMotion = self.AstroFunc.AToN(c_double(meanElem[0]))

            # print(oscElem_np)
            # Print information to output files
            if self.write_text_files:
                self.oscStateFile.write(" %17.7f%17.7f%17.7f%17.7f%17.7f%17.7f%17.7f\n" % (
                mse.value, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
                self.oscElemFile.write(" %17.7f%17.7f%17.7f%17.7f%17.7f%17.7f%17.7f\n" % (
                mse.value, oscElem[0], oscElem[1], oscElem[2], oscElem[4], oscElem[5],
                self.AstroFunc.CompTrueAnomaly(oscElem)))
                self.meanElemFile.write(" %17.7f%17.7f%17.7f%17.7f%17.7f%17.7f%17.7f\n" % (
                mse.value, meanMotion, meanElem[1], meanElem[2], meanElem[4], meanElem[5], meanElem[3]))
                self.llhElemFile.write(" %17.7f%17.7f%17.7f%17.7f%17.7f%17.7f%17.7f\n" % (
                mse.value, llh[0], llh[1], llh[2], pos[0], pos[1], pos[2]))
                self.nodalApPerFile.write(" %17.7f%17.7f%17.7f%17.7f%17.7f%17.7f%17.7f\n" % (
                mse.value, nodalApPer[0], (1440.0 / nodalApPer[0]), meanMotion, (
                        1440.0 / meanMotion), nodalApPer[1], nodalApPer[2]))

            # Check if height is below 100 km, stop propogation if so
            if llh[2] < 100.0:
                # Print an error
                if self.write_text_files:
                    if llh[2] < 0.0:
                        for outFile in self.outFiles:
                            outFile.write("Warning: Decay condition. Distance from the Geoid (Km) = %10.3f\n" % llh[2])
                    else:
                        for outFile in self.outFiles:
                            outFile.write("Warning: Height is low. HT (Km) = %10.3f\n" % llh[2])
                break  # Stop propagating
            step = step + 1

        # Remove the satellite from the propagator since we're finished with it
        if self.Sgp4Prop.Sgp4RemoveSat(key):
            ShowMsgAndTerminate(self.DllMain)


if __name__ == "__main__":
    # Standard object oriented start
    # app = Sgp4Prop('test_figures', '../input/44238_tle.inp', backtrack=True)
    # app.run()
    # app2 = Sgp4Prop('test_figures', '../input/starlink_tle_test.inp')
    # app2.run()

    app = Sgp4Prop('', '../input/44238_tle.inp', backtrack=True)
    times_dt, mean_elems, osc_elems = app.run()
    plt.plot(times_dt, mean_elems[:, 0], '.-')
    app = Sgp4Prop('', '../input/44238_tle.inp', backtrack=False)
    times_dt, mean_elems, osc_elems = app.run()
    plt.plot(times_dt, mean_elems[:, 0], '.-')
    plt.show()

    # app = Sgp4Prop('reference_sat', '../input/reference_satellite_1.inp', n_day_divs=32, end_dt=datetime(2020, 6, 1))
    # times_dt, mean_elems, osc_elems = app.run()
    # print(len(times_dt), np.shape(mean_elems))
    # fig = plt.figure()
    # for k in range(6):
    #     plt.subplot(2,3,k+1)
    #     plt.plot(times_dt, mean_elems[:, k], '.')
    #     ax = plt.gca()
    #     # plt.xlim(datetime(2020,1,1), datetime(2020,1,2))
    #     locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    #     formatter = mdates.ConciseDateFormatter(locator)
    #     ax.xaxis.set_major_locator(locator)
    #     ax.xaxis.set_major_formatter(formatter)
    #     plt.xlabel('time')
    #
    # plt.show()
