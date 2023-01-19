import os
import pyautogui
import time
import random
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets, LogLevels
from brainflow.data_filter import DataFilter


BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.ip_port = 6789
params.ip_address = "192.168.4.1"
board = BoardShim(BoardIds.CYTON_DAISY_WIFI_BOARD, params)
board.prepare_session()

#board.prepare_session()
#board.start_stream()
#BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
#time.sleep(10)
#data = board.get_board_data()
#board.stop_stream()
#board.release_session()


num_tests = 88; # 2 blocks for 44 phonemes
wait_slide = '47' # slide number of wait slide
next_slide = '48' # slide number of next slide indicator
begin_test_slide = '2' # slide number of begin test slide

initials = 'SL'

# Corresponds to slides 3-46
slides =[
    ['i_colon', 2],
    ['I', 2],
    ['hsu', 2],
    ['bar_u', 2],
    ['e', 2],
    ['rot_e', 2],
    ['uhr', 2],
    ['oa', 2],
    ['ae', 2],
    ['arch', 2],
    ['ar', 2],
    ['rot_c', 2],
    ['ie', 2],
    ['ei', 2],
    ['omega_e', 2],
    ['ci', 2],
    ['e_omega', 2],
    ['ee', 2],
    ['ai', 2],
    ['a_omega', 2],
    ['p', 2],
    ['f', 2],
    ['t', 2],
    ['theta', 2],
    ['t_integral', 2],
    ['s', 2],
    ['integral', 2],
    ['k', 2],
    ['b', 2],
    ['v', 2],
    ['d', 2],
    ['eth', 2],
    ['d3', 2],
    ['z', 2],
    ['3', 2],
    ['g', 2],
    ['h', 2],
    ['m', 2],
    ['n', 2],
    ['nn', 2],
    ['r', 2],
    ['l', 2],
    ['w', 2],
    ['j', 2],
    ]

# Tests if each block was repeated 2 times for 10 total tests per phoneme
test_slides =[
    ['i_colon', 0],
    ['I', 0],
    ['hsu', 0],
    ['bar_u', 0],
    ['e', 0],
    ['rot_e', 0],
    ['uhr', 0],
    ['oa', 0],
    ['ae', 0],
    ['arch', 0],
    ['ar', 0],
    ['rot_c', 0],
    ['ie', 0],
    ['ei', 0],
    ['omega_e', 0],
    ['ci', 0],
    ['e_omega', 0],
    ['ee', 0],
    ['ai', 0],
    ['a_omega', 0],
    ['p', 0],
    ['f', 0],
    ['t', 0],
    ['theta', 0],
    ['t_integral', 0],
    ['s', 0],
    ['integral', 0],
    ['k', 0],
    ['b', 0],
    ['v', 0],
    ['d', 0],
    ['eth', 0],
    ['d3', 0],
    ['z', 0],
    ['3', 0],
    ['g', 0],
    ['h', 0],
    ['m', 0],
    ['n', 0],
    ['nn', 0],
    ['r', 0],
    ['l', 0],
    ['w', 0],
    ['j', 0],
    ]

# Opens the powerpoint
slidenum = input()
fn = r"C:\Users\dchel\source\repos\StimPres\StimPres\stimulus.pptx"
os.startfile(fn)

time.sleep(2)

# Sets the powerpoint to fullscreen
pyautogui.hotkey('fn','f5')

time.sleep(5)



# one_block presents visual and auditory stimulus, then conducts five EEG tests separated into 2 second intervals
# slides_place is the index in the slides of the phoneme being tested
def one_block(slides_place):
    # phoneme slides begin at slide 3
    current_slide = slides_place + 3
    str_current_slide = str(current_slide)

    # keypress for the slide number, if slide number has two digits press the digit in the ones place
    pyautogui.press(str_current_slide[0])
    if(current_slide >= 10):
        pyautogui.press(str_current_slide[1])
    pyautogui.press('enter')
    
    time.sleep(3)
    
    # repeat test procedure 5 times
    for i in range(1,6):

        # test_slides incremented by 1
        test_slides[slides_place][1] += 1

        #load wait slide
        pyautogui.press(wait_slide[0])
        pyautogui.press(wait_slide[1])
        pyautogui.press('enter')

        # 1 second buffer
        time.sleep(1)

        # load phoneme slide
        pyautogui.press(str_current_slide[0])
        if(current_slide >= 10):
            pyautogui.press(str_current_slide[1])
        pyautogui.press('enter')

        
        board.start_stream()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        time.sleep(2)
        data = board.get_board_data()
        board.stop_stream()
        
        naming_convention = initials + str(test_slides[slides_place][1]) + '_' + slides[slides_place][0]
        DataFilter.write_file(data, naming_convention, 'w')

        # 2 second buffer
        #time.sleep(2)               
    

    #load slide that indicates the end of testing for current phoneme
    pyautogui.press(next_slide[0])
    pyautogui.press(next_slide[1])
    pyautogui.press('enter')

    # 3 second buffer
    time.sleep(3)

    return

# generates random phoneme and tests it if it has not reached the testing limit (10 tests for each phoneme)


while num_tests > 0:

    # pseudo-randomly generated number in range for all phonemes
    slides_place = random.randint(0, 43);

    # test the phoneme if slides counter has not reached zero
    if(slides[slides_place][1] > 0):
        slides[slides_place][1] -= 1;
        num_tests -= 1;
        one_block(slides_place)

board.release_session()
    
#prints number of blocks conducted for every phoneme
for i in range(0,43):

    print(test_slides[i][1])

