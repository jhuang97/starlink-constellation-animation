"""This module provides definitions used by the Python wrappers for the Astrodynamic Standard library.

Author: Joseph C. Lininger, A9YA
Date: January 15, 2013

"""
import sys
from ctypes import *

# Definitions for pointers to various types defined in ctypes, used for parameter type checking
# We only include definitions for those types which we actually use
# c_char_p is not defined because it's already in ctypes
c_int_p = POINTER(c_int)
c_longlong_p = POINTER(c_longlong)
c_double_p = POINTER(c_double)

def ShowMsgAndTerminate(msg = None):
   """Shows an error message, then terminates the program with an exit status of 1.

   The function takes an optional error message. If a reference to a DLL is provided, the function will assume the DLL is DllMain.dll, and will attempt to use the error facilities of the Astrodynamic Standard library to retrieve an error message. If msg is not provided or if it is explicitly set to None, the function will print a generic error message. Otherwise, msg will be printed as is.
   
   Parameters
   [msg] -- the message to display. The behavior of the function is dependent on the type of this  parameter; see above for more information.
   
   """
   # Print message from Astro Standards DLL's if a reference to DllMain.dll is provided
   if type(msg) == CDLL:
      errMsg = create_string_buffer(513)
      msg.GetLastErrMsg(errMsg)
      print( errMsg.value.rstrip() )
   # Print generic message if msg is set to None
   elif msg == None:
      print( 'ShowMsgAndTerminate() called -- No error message provided.' )
   # Print msg as is in all other cases
   else:
      print( msg )
   # Error message printed, exit with a general error status
   sys.exit(1)

def PrintWarning(softwareName, outFile):
   """Prints an export control warning to the specified output file.
   
   Parameters
   softwareName -- A string containing the name of the software program this applies to.
   outFile -- The output stream to which the message should be written.
   
   """
   
   outFile.write( '**********************************************************\n' )
   outFile.write( '*                                                        *\n' )
   outFile.write( '*                      W A R N I N G                     *\n' )
   outFile.write( '*  THIS SOFTWARE CONTAINS TECHNICAL DATA WHOSE EXPORT IS *\n' )
   outFile.write( '*  RESTRICTED BY THE ARMS EXPORT CONTROL ACT (TITLE 22,  *\n' )
   outFile.write( '*  USC, SEC 2751 ) OR EXECUTIVE ORDER 12470. VIOLATORS   *\n' )
   outFile.write( '*  OF EXPORT LAWS ARE SUBJECT TO SEVERE CRIMINAL         *\n' )
   outFile.write( '*  PENALTIES.                                            *\n' )
   outFile.write( '*                 D I S T R I B U T I O N                *\n' )
   outFile.write( '*  DISTRIBUTION IS AUTHORIZED TO US GOVERNMENT AGENCIES  *\n' )
   outFile.write( '*  AND THEIR CONTRACTORS FOR OFFICIAL USE ON A NEED TO   *\n' )
   outFile.write( '*  KNOW BASIS ONLY. ALL REQUESTS FOR THIS SOFTWARE SHALL *\n' )
   outFile.write( '*  BE REFERRED TO AFSPC/A9AC.  NO SOFTWARE CODE, MANUAL, *\n' )
   outFile.write( '*  OR MEDIA CONTAINING ANY REPRESENTATION OF THE UNITED  *\n' )
   outFile.write( '*  STATES AIR FORCE (USAF), HQ AIR FORCE SPACE COMMAND   *\n' )
   outFile.write( '*  (AFSPC) SPACE ANALYSIS CENTER (ASAC) [AFSPC/A9AC]     *\n' )
   outFile.write( '*                   %10s                           *\n' % softwareName )
   outFile.write( '*  CAPABILITY MAY BE ASSIGNED, COPIED, OR TRANSFERRED TO *\n' )
   outFile.write( '*  ANY NON-AUTHORIZED PERSON, CONTRACTOR, OR GOVERNMENT  *\n' )
   outFile.write( '*  AGENCY WITHOUT THE EXPRESSED WRITTEN CONSENT OF       *\n' )
   outFile.write( '*               USAF, HQ AFSPC/A9AC.                     *\n' )
   outFile.write( '**********************************************************\n\n' )

def CreateCArray(c_type, dimensions):
   """Creates a ctypes array with the specified dimentions.
   
   The dimensions should be specified in a list. For example, to create a 3 element array, write:
   
   [3]
   
   and to create a 3 by 2 array, write:
   
   [3, 2]

   Note, the array will be created such that accesses to it will be row-major.

   Parameters
   c_type -- A class identifier specifying the type of each element in the array. For example, c_int or c_double.
   Dimensions -- A list containing the dimensions of the array to create.
   
   Return Value
   An object representing an instance of the specified array.
   
   """
   # Create the type
   arrayType = c_type * dimensions[-1:][0]
   remainingDimensions = dimensions[:-1]
   remainingDimensions.reverse()
   for dimension in remainingDimensions:
      arrayType = arrayType * dimension
   
   # Create an instance and return it
   arrayInstance = arrayType()
   return arrayInstance

def GetIntegerField(getFunc, idx, ctype = False):
   """Retrieves an integer field from one of the data sets in the Astrodynamic Standard Library.
   
   This function can be used to retrieve things like SP application control values or TLE field values. It calls the function specified in <getFunc>, passing <idx> as the field value index. It then returns either a Python integer or a c_int, depending on the value of <ctype>.
   
   Parameters
   getFunc -- the function to call in order to retrieve a value.
   idx -- An integer corresponding to the field value to retrieve. Valid values for this will vary depending on the function being called.
   [ctype] -- If set to True, the return value will be a c_int. Otherwise a Python integer will be returned.
   
   Return Value
   The value of the requested field.
   
   """
   # Retrieve the field value
   valueStr = create_string_buffer(513)
   getFunc(idx, valueStr)
   
   # Convert the string to the requested type and return it
   if ctype:
      value = c_int(int(valueStr.value))
   else:
      value = int(valueStr.value)
   return value

def GetDoubleField(getFunc, idx, ctype = False):
   """Retrieves a floating point field (expressed as a double) from one of the data sets in the Astrodynamic Standard Library.
   
   This function can be used to retrieve things like SP application control values or TLE field values. It calls the function specified in <getFunc>, passing <idx> as the field value index. It then returns either a Python floating point value or a c_double, depending on the value of <ctype>.
   
   Parameters
   getFunc -- the function to call in order to retrieve a value.
   idx -- An integer corresponding to the field value to retrieve. Valid values for this will vary depending on the function being called.
   [ctype] -- If set to True, the return value will be a c_double. Otherwise a Python floating point value will be returned.
   
   Return Value
   The value of the requested field.
   
   """
   # Retrieve the field value
   valueStr = create_string_buffer(513)
   getFunc(idx, valueStr)
   
   # Convert the string to the requested type and return it
   if ctype:
      value = c_double(float(valueStr.value))
   else:
      value = float(valueStr.value)
   return value
