import enum

class CommandType(enum.Enum):
    # Indicates the start point of an FPGA loop.  These cannot be nested.  The
    # first 19 bits of the instruction encode the number of repetitions of
    # the loop.
    LOOP_START = 0x0

    # Wait for the computer to send a "continue" signal.  This can be used to
    # allow the computer to pause the experiment, and also to allow the computer
    # to change the DDS profiles before continuing.
    WAIT_FOR_COMPUTER = 0x1

    # Wait for a "rising edge of mains trigger" (taken from the C# description).
    # Not typically used.
    WAIT_FOR_TRIGGER = 0x2

    # A pulse which lasts for a certain number of ticks, and has the relevant
    # laser interacting.  The first 19 bits encode the number of ticks.
    FIRE_LASERS = 0x3

    # A pulse which lasts for a certain number of ticks, and has the relevant
    # laser interacting, but also receives counts from the two PMTs.  The first
    # 19 bits encode the number of ticks.
    FIRE_LASERS_AND_COUNT = 0x4

    # Cause the FPGA to finish running.  This should be the last step in any
    # experiment, but the C# can also send an "on-the-fly" stop signal before
    # this instruction is reached.
    FINISH_EXPERIMENT = 0x5

    # Send the data stored in the FPGA's SRAM to the computer.  The SRAM is
    # 512kB, and each `FIRE_LASERS_AND_COUNT` operation causes 6 bytes to be
    # stored in it (4 bytes for the count, and 2 bytes for the error state).
    SEND_DATA = 0x6

    # Indicates the end point of an FPGA loop.  These cannot be nested.  After
    # this instruction, control passes to the instruction which immediately
    # succeeds the most recent `LOOP_START` instruction.
    LOOP_END = 0x7

def instruction(length, lasers, dds_profile, type):
    """
    Instruction layout:
                                   854
                               DDS1  | b2
                           radial |  | |b1
                           DDS2 | |  | ||
                              | | |  | ||
        |........|........|........|........|
         |-------------------| | |  | |  |-|
              length        aux1 |  | |  type
                              trap  | |
                                 DDS0 |
                                    729
    """
    return bytes((
        (length & 0x0007_f800) >> 11,

        (length & 0x0000_07f8) >> 3,

        ((length & 0x0000_0007) << 5)\
        | (((dds_profile & 0x4) ^ 0x4) << 2)\
        | (("aux1" in lasers) << 3)\
        | (("radial" in lasers) << 2)\
        | (("trap" in lasers) << 1)\
        | (((dds_profile & 0x2) ^ 0x2) >> 1),

        (((dds_profile & 0x1) ^ 0x1) << 7)\
        | (("854" in lasers) << 6)\
        | (("729" in lasers) << 5)\
        | (("b2" in lasers) << 4)\
        | (("b1" in lasers) << 3)\
        | type.value,
    ))

STOP = instruction(0, set(), 0, CommandType.FINISH_EXPERIMENT)
