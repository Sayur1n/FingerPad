# -----------------------------------------------------------------------------
#  Copyright (c) 2015 Pressure Profile Systems
#
#  Licensed under the MIT license. This file may not be copied, modified, or
#  distributed except according to those terms.
# -----------------------------------------------------------------------------

class SerialCommand:
    TIMEOUT = 100
    TIMESTAMP_SIZE = 4

    @staticmethod
    def GenerateWriteCommand(i2cAddress, ID, writeLocation, data):
        """
        Generate raw command packet
        
        Args:
            i2cAddress: I2C address
            ID: Command ID
            writeLocation: Write location
            data: Data to write
            
        Returns:
            Raw command packet as byte array
        """
        command = bytearray(len(data) + 15)
        
        for i in range(4):
            command[i] = 0xFF
        
        command[4] = i2cAddress
        command[5] = SerialCommand.TIMEOUT
        command[6] = ID
        command[7] = 0x02
        command[8] = writeLocation
        command[9] = len(data)
        command[10 + len(data)] = 0xFF
        
        for i in range(len(data)):
            command[10 + i] = data[i]
        
        for i in range(4):
            command[11 + i + len(data)] = 0xFE
        
        return bytes(command)

    @staticmethod
    def GenerateWriteCalCommand(i2cAddress, ID, writeLocation, data):
        """
        Generate raw command packet
        
        Args:
            i2cAddress: I2C address
            ID: Command ID
            writeLocation: Write location
            data: Data to write
            
        Returns:
            Raw command packet as byte array
        """
        command = bytearray(len(data) + 15)
        
        for i in range(4):
            command[i] = 0xFF
        
        command[4] = i2cAddress
        command[5] = SerialCommand.TIMEOUT
        command[6] = ID
        command[7] = 0x04
        command[8] = writeLocation
        command[9] = len(data)
        command[10 + len(data)] = 0xFF
        
        for i in range(len(data)):
            command[10 + i] = data[i]
        
        for i in range(4):
            command[11 + i + len(data)] = 0xFE
        
        return bytes(command)

    @staticmethod
    def GenerateReadCommand(i2cAddress, ID, readLocation, numToRead):
        """
        Generate raw command packet
        
        Args:
            i2cAddress: I2C address
            ID: Command ID
            readLocation: Read location
            numToRead: Number of bytes to read
            
        Returns:
            Raw command packet as byte array
        """
        command = bytearray(16)
        # print(command)
        
        for i in range(4):
            command[i] = 0xFF
        
        command[4] = i2cAddress
        command[5] = SerialCommand.TIMEOUT
        command[6] = ID
        command[7] = 0x01
        command[8] = readLocation
        command[9] = numToRead
        command[10] = 0xFF
        
        for i in range(4):
            command[11 + i] = 0xFE
        # print(command)
            
        # result = command.replace(b'd', b'')
        # print(command)
        # print(result)
        
        return bytes(command)

    @staticmethod
    def GenerateToggleCommand(i2cAddress, ID, writeLocation, data):
        """
        Generate raw command packet
        
        Args:
            i2cAddress: I2C address
            ID: Command ID
            writeLocation: Write location
            data: Data to write
            
        Returns:
            Raw command packet as byte array
        """
        command = bytearray(16 + 15)
        
        for i in range(4):
            command[i] = 0xFF
        
        command[4] = i2cAddress
        command[5] = SerialCommand.TIMEOUT
        command[6] = ID
        command[7] = 0x03
        command[8] = data
        command[9] = 16
        command[10 + 16] = 0xFF
        
        for i in range(16):
            command[10 + i] = 0x07
        
        for i in range(4):
            command[11 + i + 16] = 0xFE
        
        return bytes(command)
