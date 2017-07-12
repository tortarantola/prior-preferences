#!/usr/bin/env python2
# -*- coding: utf-8 -*-

######################## CITATIONS ########################## 
##### Binary choice, BDM, auction routines, and instructions are modified from:
##### 
##### De Martino, B., Fleming, S. M., Garrett, N., & Dolan, R. J. (2012). Confidence in value-based choice. Nature Neuroscience, 16(1), 105-110. 
#####
##### Food item images are original.

"""
This experiment was created using PsychoPy2 Experiment Builder (v1.80.01), July 16, 2014, at 03:20
If you publish work using this script please cite the relevant PsychoPy publications
  Peirce, JW (2007) PsychoPy - Psychophysics software in Python. Journal of Neuroscience Methods, 162(1-2), 8-13.
  Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy. Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, core, data, event, logging, sound, gui
from psychopy.constants import *  # things bdm STARTED, FINISHED
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
from pyglet.window import key # to detect key state, whether key is held down, to move slider on key hold
import pandas as pd
import datetime
from psychopy.iohub import launchHubServer
import pylink

# Initialize IOHub for eye tracker
# io=launchHubServer(iohub_config_name='iohub_config.yaml')
# tracker = io.devices.tracker

# Store info about the experiment session
expName = 'infer_design1_no_eyetracking'  # from the Builder filename that created this script
expInfo = {u'session': u'001', u'participant': u'', u'eye': u'', u'glasses': u'', u'contacts': u''}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Setup filename for saving
filename = 'data/%s_%s_%s' %(expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
#save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# set up variable to track current state of key press, to move slider when keys held down
keyState=key.KeyStateHandler()

# Setup the Window
screenwidth = 1920
screenheight = 1080
win = visual.Window(size=(screenwidth, screenheight), fullscr=True, screen=0, allowGUI=False, allowStencil=False,
    monitor='testMonitor', color='black', colorSpace='rgb',
    blendMode='avg', useFBO=True,
    )
win.winHandle.push_handlers(keyState)


# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win.getActualFrameRate()
if expInfo['frameRate']!=None:
    frameDur = 1.0/round(expInfo['frameRate'])
else:
    frameDur = 1.0/60.0 # couldn't get a reliable measure so guess


######################## COMPONENTS ##########################

# Set up directory path for stimuli. This avoids us having to put the entire image path in the conditions spreadsheet.
dir_path = 'stimuli/itempics/modified/small/'

# Initialize components for Routine "instr_main"
instr_mainClock = core.Clock()
instr_main_txt = visual.TextStim(win=win, ori=0, name='instr_main_txt',
    text=u'Welcome!\n\nPlease read the instructions carefully and get the experimenter when you\'re finished. Take as much time as you need.',    font=u'Arial',
    pos=[0, 0], height=0.06, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "instr_bdm"
instr_bdmClock = core.Clock()
instr_bdm_txt = visual.TextStim(win=win, ori=0, name='instr_bdm_txt',
    text=u'The bidding task is about to begin. Use the LEFT and RIGHT arrow keys to move the cursor along the slider, then press the DOWN arrow to enter your bid.\n\n[press space bar to begin]',    font=u'Arial',
    pos=[0, 0], height=0.08, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "fixation"
fixationClock = core.Clock()
fixation_text = visual.TextStim(win=win, ori=0, name='fixation_text',
    text=u'+',    font=u'Arial',
    pos=[0, 0], height=0.1, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "bdm"
bdmClock = core.Clock()
bdm_pic = visual.ImageStim(win=win, name='bdm_pic',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=None,
    color=[1,1,1], colorSpace=u'rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=False, depth=0.0)
bdm_bid = visual.RatingScale(win=win, name='bdm_bid', marker=u'triangle', markerColor=u'orange', leftKeys=None, rightKeys=None,
    size=1.0, pos=[0.0, -0.6], low=0, high=3, precision=100, labels=[u'\xa30', u'\xa33'],
    scale=u'', markerStart=u'1.5', tickHeight=u'1', showAccept=False, acceptKeys=[u'down', u'return'])

# Initialize components for Routine "instr_choice"
instr_choiceClock = core.Clock()
instr_choice_txt = visual.TextStim(win=win, ori=0, name='instr_choice_txt',
    text=u'The choice task is about to begin. Choose your preferred item by pressing the LEFT or RIGHT arrow keys, then use the slider to indicate how confident you are that you made the best choice.\n\n[press space bar to begin]',    font=u'Arial',
    pos=[0, 0], height=0.08, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "choice"
choiceClock = core.Clock()
choice_pic_left = visual.ImageStim(win=win, name='choice_pic_left',
    image='sin', mask=None,
    ori=0, pos=[-0.5, 0], size=None,
    color=[1,1,1], colorSpace=u'rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=False, depth=0.0)
choice_pic_right = visual.ImageStim(win=win, name='choice_pic_right',
    image='sin', mask=None,
    ori=0, pos=[0.5, 0], size=None,
    color=[1,1,1], colorSpace=u'rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=False, depth=-1.0)

# Initialize components for Routine "choice_selection"
choice_selectionClock = core.Clock()
star_left_selection = visual.TextStim(win=win, ori=0, name='star_left_selection',
    text=u'*',    font=u'Arial',
    pos=[-0.5, -0.8], height=0.5, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=-2.0)
star_right_selection = visual.TextStim(win=win, ori=0, name='star_right_selection',
    text=u'*',    font=u'Arial',
    pos=[0.5, -0.8], height=0.5, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=-3.0)

# Initialize components for Routine "confidence"
confidenceClock = core.Clock()
confidence_rating = visual.RatingScale(win=win, name='confidence_rating', marker=u'triangle', markerColor=u'orange', leftKeys=None, rightKeys=None,
    size=1.0, pos=[0.0, 0.0], low=1, high=6, precision=20,
    scale=u'', markerStart=u'3.5', tickHeight=u'1', showAccept=False, acceptKeys=[u'down', u'return'])

# Initialize components for Routine "instr_infer_intro"
instr_infer_introClock = core.Clock()
instr_infer_intro_txt = visual.TextStim(win=win, ori=0, name='instr_infer_intro_txt',
    text=u'Thanks! Now please get the experimenter, who will give you the instructions for the second part of the experiment. \n\nPlease read these instructions carefully and get the experimenter again when you\'re finished. Take as much time as you need.',    font=u'Arial',
    pos=[0, 0], height=0.06, wrapWidth=1.5,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "instr_infer_practice"
instr_infer_practiceClock = core.Clock()
instr_infer_practice_txt = visual.TextStim(win=win, ori=0, name='instr_infer_practice_txt',
    text=u'First, you\'ll do some practice trials. These are just for you to get used to the task; your responses won\'t count. \n\n[press space bar to begin]',    font=u'Arial',
    pos=[0, 0], height=0.06, wrapWidth=1.5,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "instr_infer"
instr_inferClock = core.Clock()
instr_infer_txt = visual.TextStim(win=win, ori=0, name='instr_infer_txt',
    text=u'Great! Now that you\'ve done some practice trials, we\'re ready to start the real trials. Please call the experimenter into the room.',    font=u'Arial',
    pos=[0, 0], height=0.06, wrapWidth=1.5,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "get_ready"
get_readyClock = core.Clock()
get_ready_text = visual.TextStim(win=win, ori=0, name='get_ready_text',
    text=u'Get ready!',    font=u'Arial',
    pos=[0, 0], height=0.12, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "trial"
trialClock = core.Clock()
img_left_infer = visual.ImageStim(win=win, name='img_left_infer',
    image='sin', mask=None,
    ori=0, pos=[-0.5, 0], size=None,
    color=[1,1,1], colorSpace=u'rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=False, depth=0.0)
img_right_infer = visual.ImageStim(win=win, name='img_right_infer',
    image='sin', mask=None,
    ori=0, pos=[0.5, 0], size=None,
    color=[1,1,1], colorSpace=u'rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=False, depth=-1.0)

# Initialize components for Routine "trial_post_response"
trial_post_responseClock = core.Clock()
selection_arrow_left = visual.TextStim(win=win, ori=0, name='selection_arrow_left',
    text=u'*',    font=u'Arial',
    pos=[-0.5, -0.09], height=0.5, wrapWidth=None,
    color=u'red', colorSpace=u'rgb', opacity=1,
    depth=-2.0)
selection_arrow_right = visual.TextStim(win=win, ori=0, name='selection_arrow_right',
    text=u'*',    font=u'Arial',
    pos=[0.5, -0.09], height=0.5, wrapWidth=None,
    color=u'red', colorSpace=u'rgb', opacity=1,
    depth=-3.0)
feedback_box_left = visual.Rect(win=win, name='feedback_box_left',
    width=[0.9, 0.9][0], height=[0.9, 0.9][1],
    ori=0, pos=[-0.5, 0],
    lineWidth=5, lineColor=u'yellow', lineColorSpace=u'rgb',
    fillColor=None, fillColorSpace=u'rgb',
    opacity=1, depth=-4.0, interpolate=True)
feedback_box_right = visual.Rect(win=win, name='feedback_box_right',
    width=[0.9, 0.9][0], height=[0.9, 0.9][1],
    ori=0, pos=[0.5, 0],
    lineWidth=5, lineColor=u'yellow', lineColorSpace=u'rgb',
    fillColor=None, fillColorSpace=u'rgb',
    opacity=1, depth=-5.0, interpolate=True)

# Initialize components for Routine "rest_prompt"
rest_promptClock = core.Clock()
rest_prompt_txt = visual.TextStim(win=win, ori=0, name='rest_prompt_txt',
    text=u'Great! Now take a rest and press the space bar when you\u2019re ready to begin the next block.',    font=u'Arial',
    pos=[0, 0], height=0.08, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Create counters for the number of correct and incorrect guesses made by subjects to determine extra payment.
correct_counter = 0
incorrect_counter = 0

# Create list of lists to store binary choice and BDM data to generate food reward at the end of the experiment
prefs = [['left', 'right', 'choice', 'bid']]
bids = [['item', 'bid']] # For the bids collected during the first BDM routine, at the beginning of the experiment

### CHOICE & INFERENCE RANDOMIZATION ###

# Define two functions to generate constrained pseudorandom sequences of item pair presentations for the binary choice and inference
# parts of the task. The constraint ensures that the same pair of items does not appear twice in a row (flipped or not).
# In the conditions spreadsheet that the item pairs are drawn from, the flipped version of the same item pair is 20
# rows apart.

# Binary choice randomization
def genseq_choice():
    done = False
    while done==False:
        counter = 0
        ans = np.random.choice(40, 40, replace=False) # Generate a pseudorandom sequence of 40 numbers
        for x in range(1,40):
            if (ans[x]-ans[x-1]) % 20 == 0: # Is the difference between any sequential numbers divisible by 20?
                counter += 1
        if counter==0: # If not, end the loop and return the array
            done = True
            return ans
        else: # If so, continue the loop and try again
            done = False

# Inference practice block randomization
def genseq_infer_practice():
    done = False
    while done==False:
        counter = 0
        ans = np.random.choice(14, 14, replace=False) # Generate a pseudorandom sequence of 200 numbers
        for x in range(1,14):
            if (ans[x]-ans[x-1]) % 7 == 0: # Is the difference between any sequential numbers divisible by 20?
                counter += 1
        if counter==0: # If not, end the loop and return the array
            done = True
            return ans
        else: # If so, continue the loop and try again
            done = False

# Inference randomization
def genseq_infer():
    done = False
    while done==False:
        counter = 0
        ans = np.random.choice(200, 200, replace=False) # Generate a pseudorandom sequence of 200 numbers
        for x in range(1,200):
            if (ans[x]-ans[x-1]) % 20 == 0: # Is the difference between any sequential numbers divisible by 20?
                counter += 1
        if counter==0: # If not, end the loop and return the array
            done = True
            return ans
        else: # If so, continue the loop and try again
            done = False


# Set independent sequences for each of the two choice blocks, the inference practice block, and the three inference rest blocks. Each rest block contains 10 presentations of
# each pair.

choiceseq = genseq_choice()
practiceseq = genseq_infer_practice()
block1seq = genseq_infer()
block2seq = genseq_infer()
block3seq = genseq_infer()


# Choose a pilot participant whose choices will form the basis of the learning task
partners = ['P1','P2','P3','P4','P5','P6','P8','P9','P10','P11','P12'] # List of pilot participants, P7 excluded for perfectly inconsistent choices
partner = np.random.choice(partners,1) # Choose one at random
partner_file = 'conditions/choices/infer_design1_value_pairs_' + partner[0] + '.csv' # Path to that participant's choice data

# Create separate CSV files, one for each choice, practice, and rest block, with the item pairs in the pseudorandomly generated order.
# These CSV files will be used as PsychoPy's conditions spreadsheets and run sequentially in the choice, practice, and inference loops below.


choicecond_src = pd.read_csv('conditions/choice_design1_binary.csv', index_col=None, header=0)
choicecond = choicecond_src.copy()
for x in range(40):
    seq = choiceseq[x]
    choicecond.iloc[x] = choicecond_src.iloc[seq]
choicecond.to_csv(path_or_buf=filename+'_choicecond.csv', index=False)

practicecond_src = pd.read_csv('conditions/choices/infer_design1_value_pairs_practice.csv', index_col=None, header=0)
practicecond = practicecond_src.copy()
for x in range(14):
    seq = practiceseq[x]
    practicecond.iloc[x] = practicecond_src.iloc[seq]
practicecond.to_csv(path_or_buf=filename+'_practicecond.csv', index=False)

block1cond_src = pd.read_csv(partner_file, index_col=None, header=0)
block1cond = block1cond_src.copy()
for x in range(200):
    seq = block1seq[x]
    block1cond.iloc[x] = block1cond_src.iloc[seq]
block1cond.to_csv(path_or_buf=filename+'_block1cond.csv', index=False)

block2cond_src = pd.read_csv(partner_file, index_col=None, header=0)
block2cond = block2cond_src.copy()
for x in range(200):
    seq = block2seq[x]
    block2cond.iloc[x] = block2cond_src.iloc[seq]
block2cond.to_csv(path_or_buf=filename+'_block2cond.csv', index=False)

block3cond_src = pd.read_csv(partner_file, index_col=None, header=0)
block3cond = block3cond_src.copy()
for x in range(200):
    seq = block3seq[x]
    block3cond.iloc[x] = block3cond_src.iloc[seq]
block3cond.to_csv(path_or_buf=filename+'_block3cond.csv', index=False)


# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

#------Prepare to start Routine "instr_main"-------
t = 0
instr_mainClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instr_main_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
instr_main_resp.status = NOT_STARTED
# keep track of which components have finished
instr_mainComponents = []
instr_mainComponents.append(instr_main_txt)
instr_mainComponents.append(instr_main_resp)
for thisComponent in instr_mainComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr_main"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instr_mainClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_main_txt* updates
    if t >= 0.0 and instr_main_txt.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_main_txt.tStart = t  # underestimates by a little under one frame
        instr_main_txt.frameNStart = frameN  # exact frame index
        instr_main_txt.setAutoDraw(True)
    
    # *instr_main_resp* updates
    if t >= 5.0 and instr_main_resp.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_main_resp.tStart = t  # underestimates by a little under one frame
        instr_main_resp.frameNStart = frameN  # exact frame index
        instr_main_resp.status = STARTED
        # keyboard checking is just starting
        instr_main_resp.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if instr_main_resp.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            instr_main_resp.keys = theseKeys[-1]  # just the last key pressed
            instr_main_resp.rt = instr_main_resp.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_mainComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr_main"-------
for thisComponent in instr_mainComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instr_main_resp.keys in ['', [], None]:  # No response was made
   instr_main_resp.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('instr_main_resp.keys',instr_main_resp.keys)
if instr_main_resp.keys != None:  # we had a response
    thisExp.addData('instr_main_resp.rt', instr_main_resp.rt)
thisExp.nextEntry()



#------Prepare to start Routine "instr_bdm"-------
t = 0
instr_bdmClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instr_bdm_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
instr_bdm_resp.status = NOT_STARTED
# keep track of which components have finished
instr_bdmComponents = []
instr_bdmComponents.append(instr_bdm_txt)
instr_bdmComponents.append(instr_bdm_resp)
for thisComponent in instr_bdmComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr_bdm"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instr_bdmClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_bdm_txt* updates
    if t >= 0.0 and instr_bdm_txt.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_bdm_txt.tStart = t  # underestimates by a little under one frame
        instr_bdm_txt.frameNStart = frameN  # exact frame index
        instr_bdm_txt.setAutoDraw(True)
    
    # *instr_bdm_resp* updates
    if t >= 2.0 and instr_bdm_resp.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_bdm_resp.tStart = t  # underestimates by a little under one frame
        instr_bdm_resp.frameNStart = frameN  # exact frame index
        instr_bdm_resp.status = STARTED
        # keyboard checking is just starting
        instr_bdm_resp.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if instr_bdm_resp.status == STARTED:
        theseKeys = event.getKeys(keyList=['space', 's'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            instr_bdm_resp.keys = theseKeys[-1]  # just the last key pressed
            instr_bdm_resp.rt = instr_bdm_resp.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_bdmComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr_bdm"-------
for thisComponent in instr_bdmComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instr_bdm_resp.keys in ['', [], None]:  # No response was made
   instr_bdm_resp.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('instr_bdm_resp.keys',instr_bdm_resp.keys)
thisExp.addData('partnered_pilot_participant', partner)
if instr_bdm_resp.keys != None:  # we had a response
    thisExp.addData('instr_bdm_resp.rt', instr_bdm_resp.rt)
thisExp.nextEntry()


######################## BDM LOOP 1 ##########################


# set up handler to look after randomisation of conditions etc
bdm_loop1 = data.TrialHandler(nReps=1, method=u'random', 
    extraInfo=expInfo, originPath=None,
    trialList=data.importConditions(u'conditions/choice_design1_bdm.xlsx'),
    seed=None, name='bdm_loop1')
thisExp.addLoop(bdm_loop1)  # add the loop to the experiment
thisbdm_loop1 = bdm_loop1.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisbdm_loop1.rgb)
if thisbdm_loop1 != None:
    for paramName in thisbdm_loop1.keys():
        exec(paramName + '= thisbdm_loop1.' + paramName)

# Check if the 'skip' key was pressed in the instructions routine; if so, end the loop and move on to the inference task
if instr_bdm_resp.keys=='s':
    bdm_loop1.finished = True

for thisbdm_loop1 in bdm_loop1:
    currentLoop = bdm_loop1
    # abbreviate parameter names if possible (e.g. rgb = thisbdm_loop1.rgb)
    if thisbdm_loop1 != None:
        for paramName in thisbdm_loop1.keys():
            exec(paramName + '= thisbdm_loop1.' + paramName)
    
    #------Prepare to start Routine "bdm"-------
    t = 0
    bdmClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    bdm_pic.setImage(dir_path+bdm_img)
    bdm_bid.reset()
    # jitter the starting position of the BDM scale from a uniform distribution between 1 and 2, rounded to the nearest decimal place
    bdm_marker_start = float(np.round(np.random.uniform(1, 2, 1)[0], 1))
    bdm_bid.setMarkerPos(bdm_marker_start)
    # keep track of which components have finished
    bdmComponents = []
    bdmComponents.append(fixation_text)
    bdmComponents.append(bdm_pic)
    bdmComponents.append(bdm_bid)
    for thisComponent in bdmComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "bdm"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = bdmClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_text* updates
        if t >= 0 and fixation_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            fixation_text.tStart = t  # underestimates by a little under one frame
            fixation_text.frameNStart = frameN  # exact frame index
            fixation_text.setAutoDraw(True)
        elif fixation_text.status == STARTED and t>= (0 + (1.0-win.monitorFramePeriod*0.75)):
            fixation_text.setAutoDraw(False)

        # *bdm_pic* updates
        if t >= 1.0 and bdm_pic.status == NOT_STARTED:
            # keep track of start time/frame for later
            bdm_pic.tStart = t  # underestimates by a little under one frame
            bdm_pic.frameNStart = frameN  # exact frame index
            bdm_pic.setAutoDraw(True)
        # *bdm_bid* updates
        if t >= 1.0:
            bdm_bid.draw()
            continueRoutine = bdm_bid.noResponse
            if bdm_bid.noResponse == False:
                bdm_bid.response = bdm_bid.getRating()
                bdm_bid.rt = bdm_bid.getRT()
            elif bdm_bid.noResponse==True:
                if keyState[key.LEFT]==True and bdm_bid.markerPlacedAt >0.01:
                    bdm_bid.markerPlacedAt = bdm_bid.markerPlacedAt - 0.02
                    bdm_bid.draw()
                elif keyState[key.LEFT]==True and bdm_bid.markerPlacedAt==0.01:
                    bdm_bid.markerPlacedAt = bdm_bid.markerPlacedAt - 0.01
                    bdm_bid.draw()
                elif keyState[key.RIGHT]==True and bdm_bid.markerPlacedAt <2.99:
                    bdm_bid.markerPlacedAt = bdm_bid.markerPlacedAt + 0.02
                    bdm_bid.draw()
                elif keyState[key.RIGHT]==True and bdm_bid.markerPlacedAt==2.99:
                    bdm_bid.markerPlacedAt = bdm_bid.markerPlacedAt + 0.01
                    bdm_bid.draw()
                    
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in bdmComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()
    
    #-------Ending Routine "bdm"-------
    for thisComponent in bdmComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store data for bdm (TrialHandler)
    bdm_loop1.addData('bdm_bid1.response', bdm_bid.getRating())
    bdm_loop1.addData('bdm_bid1.rt', bdm_bid.getRT())
    bdm_loop1.addData('bdm_marker_start', bdm_marker_start)

    # Add the item and the subject's bid to the 'bids' list, which will be merged with the 'prefs' list later in order to run the auction
    bids.append([bdm_img, bdm_bid.response])
    
    thisExp.nextEntry()
    
# completed 1 repeat of 'bdm_loop1'


#------Prepare to start Routine "instr_choice"-------
t = 0
instr_choiceClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instr_choice_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
instr_choice_resp.status = NOT_STARTED
# keep track of which components have finished
instr_choiceComponents = []
instr_choiceComponents.append(instr_choice_txt)
instr_choiceComponents.append(instr_choice_resp)
for thisComponent in instr_choiceComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr_choice"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instr_choiceClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_choice_txt* updates
    if t >= 0.0 and instr_choice_txt.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_choice_txt.tStart = t  # underestimates by a little under one frame
        instr_choice_txt.frameNStart = frameN  # exact frame index
        instr_choice_txt.setAutoDraw(True)
    
    # *instr_choice_resp* updates
    if t >= 2.0 and instr_choice_resp.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_choice_resp.tStart = t  # underestimates by a little under one frame
        instr_choice_resp.frameNStart = frameN  # exact frame index
        instr_choice_resp.status = STARTED
        # keyboard checking is just starting
        instr_choice_resp.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if instr_choice_resp.status == STARTED:
        theseKeys = event.getKeys(keyList=['space', 's'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            instr_choice_resp.keys = theseKeys[-1]  # just the last key pressed
            instr_choice_resp.rt = instr_choice_resp.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_choiceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr_choice"-------
for thisComponent in instr_choiceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instr_choice_resp.keys in ['', [], None]:  # No response was made
   instr_choice_resp.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('instr_choice_resp.keys',instr_choice_resp.keys)
if instr_choice_resp.keys != None:  # we had a response
    thisExp.addData('instr_choice_resp.rt', instr_choice_resp.rt)
thisExp.nextEntry()


######################## BINARY LOOP ##########################

# set up handler to look after randomisation of conditions etc
binary = data.TrialHandler(nReps=1, method=u'sequential', 
    extraInfo=expInfo, originPath=None,
    trialList=data.importConditions(filename+'_choicecond.csv'),
    seed=None, name='binary')
thisExp.addLoop(binary)  # add the loop to the experiment
thisBinary1 = binary.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisBinary1.rgb)
if thisBinary1 != None:
    for paramName in thisBinary1.keys():
        exec(paramName + '= thisBinary1.' + paramName)

# Check if the 'skip' key was pressed in the instructions routine; if so, end the loop and move on to the BDM
if instr_choice_resp.keys=='s':
    binary.finished = True

for thisBinary1 in binary:
    currentLoop = binary
    # abbreviate parameter names if possible (e.g. rgb = thisBinary1.rgb)
    if thisBinary1 != None:
        for paramName in thisBinary1.keys():
            exec(paramName + '= thisBinary1.' + paramName)
    
    #------Prepare to start Routine "choice"-------
    t = 0
    choiceClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    choice_pic_left.setImage(dir_path+choice_left)
    choice_pic_right.setImage(dir_path+choice_right)
    event.clearEvents(eventType='keyboard') # Clear keyboard event log to avoid last key press from prior loop being used
    key_resp_choice = event.BuilderKeyResponse()  # create an object of type KeyResponse
    key_resp_choice.status = NOT_STARTED
    # keep track of which components have finished
    choiceComponents = []
    choiceComponents.append(fixation_text)
    choiceComponents.append(choice_pic_left)
    choiceComponents.append(choice_pic_right)
    choiceComponents.append(key_resp_choice)
    for thisComponent in choiceComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    
    #-------Start Routine "choice"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = choiceClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_text* updates
        if t >= 0 and fixation_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            fixation_text.tStart = t  # underestimates by a little under one frame
            fixation_text.frameNStart = frameN  # exact frame index
            fixation_text.setAutoDraw(True)
        elif fixation_text.status == STARTED and t>= (0 + (1.0-win.monitorFramePeriod*0.75)):
            fixation_text.setAutoDraw(False)

        # *choice_pic_left* updates
        if t >= 1.0 and choice_pic_left.status == NOT_STARTED:
            # keep track of start time/frame for later
            choice_pic_left.tStart = t  # underestimates by a little under one frame
            choice_pic_left.frameNStart = frameN  # exact frame index
            choice_pic_left.setAutoDraw(True)
        
        # *choice_pic_right* updates
        if t >= 1.0 and choice_pic_right.status == NOT_STARTED:
            # keep track of start time/frame for later
            choice_pic_right.tStart = t  # underestimates by a little under one frame
            choice_pic_right.frameNStart = frameN  # exact frame index
            choice_pic_right.setAutoDraw(True)

        # *key_resp_choice* updates
        if t >= 1.0 and key_resp_choice.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_resp_choice.tStart = t  # underestimates by a little under one frame
            key_resp_choice.frameNStart = frameN  # exact frame index
            key_resp_choice.status = STARTED
            # keyboard checking is just starting
            key_resp_choice.clock.reset()  # now t=0
        if key_resp_choice.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                key_resp_choice.keys = theseKeys[-1]  # just the last key pressed
                key_resp_choice.rt = key_resp_choice.clock.getTime()
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in choiceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()
    
    #-------Ending Routine "choice"-------
    for thisComponent in choiceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_resp_choice.keys in ['', [], None]:  # No response was made
       key_resp_choice.keys=None
    # store data for binary (TrialHandler)
    binary.addData('key_resp_choice.keys',key_resp_choice.keys)
    if key_resp_choice.keys != None:  # we had a response
        binary.addData('key_resp_choice.rt', key_resp_choice.rt)

    # define a function to append binary choices and the bid for the chosen item to the preference list
    # lft is the left image displayed during that choice trial
    # rt is the right image displayed
    def rec_choice(lft, rt): 
        bd = False
        if key_resp_choice.keys=='left':
            chc = lft
        elif key_resp_choice.keys=='right':
            chc = rt
        for x in range(len(bids)):
            if bids[x][0]==chc:
                bd = bids[x][1]
        prefs.append([lft, rt, chc, bd])

    # call the function
    rec_choice(choice_left, choice_right)
    

    #------Prepare to start Routine "choice_selection"-------
    t = 0
    choice_selectionClock.reset()  # clock 
    frameN = -1
    routineTimer.add(1.000000)
    # keep track of which components have finished
    choice_selectionComponents = []
    choice_selectionComponents.append(choice_pic_left)
    choice_selectionComponents.append(choice_pic_right)
    choice_selectionComponents.append(star_left_selection)
    choice_selectionComponents.append(star_right_selection)
    for thisComponent in choice_selectionComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "choice_selection"-------
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = choice_selectionClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *choice_pic_left* updates
        if t >= 0.0 and choice_pic_left.status == NOT_STARTED:
            # keep track of start time/frame for later
            choice_pic_left.tStart = t  # underestimates by a little under one frame
            choice_pic_left.frameNStart = frameN  # exact frame index
            choice_pic_left.setAutoDraw(True)
        elif choice_pic_left.status == STARTED and t >= (0.0 + (1.0-win.monitorFramePeriod*0.75)): #most of one frame period left
            choice_pic_left.setAutoDraw(False)
        
        # *choice_pic_right* updates
        if t >= 0.0 and choice_pic_right.status == NOT_STARTED:
            # keep track of start time/frame for later
            choice_pic_right.tStart = t  # underestimates by a little under one frame
            choice_pic_right.frameNStart = frameN  # exact frame index
            choice_pic_right.setAutoDraw(True)
        elif choice_pic_right.status == STARTED and t >= (0.0 + (1.0-win.monitorFramePeriod*0.75)): #most of one frame period left
            choice_pic_right.setAutoDraw(False)
        
        # *star_left_selection* updates
        if key_resp_choice.keys=='left':
            if t >= 0.0 and star_left_selection.status == NOT_STARTED:
                # keep track of start time/frame for later
                star_left_selection.tStart = t  # underestimates by a little under one frame
                star_left_selection.frameNStart = frameN  # exact frame index
                star_left_selection.setAutoDraw(True)
            elif star_left_selection.status == STARTED and t >= (0.0 + (1.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                star_left_selection.setAutoDraw(False)
        
        # *star_right_selection* updates
        if key_resp_choice.keys=='right':
            if t >= 0.0 and star_right_selection.status == NOT_STARTED:
                # keep track of start time/frame for later
                star_right_selection.tStart = t  # underestimates by a little under one frame
                star_right_selection.frameNStart = frameN  # exact frame index
                star_right_selection.setAutoDraw(True)
            elif star_right_selection.status == STARTED and t >= (0.0 + (1.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                star_right_selection.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in choice_selectionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "choice_selection"-------
    for thisComponent in choice_selectionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    #thisExp.nextEntry()

    #------Prepare to start Routine "confidence"-------
    t = 0
    confidenceClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    confidence_rating.reset()
    # jitter the starting position of the confidence rating scale from a uniform distribution between 1.7 and 3.3 (the middle third of the scale), rounded to the nearest decimal place
    confidence_marker_start = float(np.round(np.random.uniform(1.7, 3.3, 1)[0], 1))
    confidence_rating.setMarkerPos(confidence_marker_start)
    # keep track of which components have finished
    confidenceComponents = []
    confidenceComponents.append(confidence_rating)
    for thisComponent in confidenceComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "confidence"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = confidenceClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *confidence_rating* updates
        if t > 0.5:
            confidence_rating.draw()
            continueRoutine = confidence_rating.noResponse
            if confidence_rating.noResponse == False:
                confidence_rating.response = confidence_rating.getRating()
                confidence_rating.rt = confidence_rating.getRT()
            elif confidence_rating.noResponse==True:
                if keyState[key.LEFT]==True and confidence_rating.markerPlacedAt >0:
                    confidence_rating.markerPlacedAt = confidence_rating.markerPlacedAt - 0.1
                    confidence_rating.draw()
                elif keyState[key.LEFT]==True and confidence_rating.markerPlacedAt==0.1:
                    confidence_rating.markerPlacedAt = confidence_rating.markerPlacedAt - 0.1
                    confidence_rating.draw()
                elif keyState[key.RIGHT]==True and confidence_rating.markerPlacedAt <4.9:
                    confidence_rating.markerPlacedAt = confidence_rating.markerPlacedAt + 0.1
                    confidence_rating.draw()
                elif keyState[key.RIGHT]==True and confidence_rating.markerPlacedAt==4.9:
                    confidence_rating.markerPlacedAt = confidence_rating.markerPlacedAt + 0.1
                    confidence_rating.draw()
                    
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in confidenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()
    
    #-------Ending Routine "confidence"-------
    for thisComponent in confidenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store data for confidence (TrialHandler)
    binary.addData('confidence_rating1.response', confidence_rating.getRating())
    binary.addData('confidence_rating1.rt', confidence_rating.getRT())
    binary.addData('confidence_marker_start', confidence_marker_start+1) # Add 1 to convert from 0-5 to 1-6 scale

    thisExp.nextEntry()


# completed 1 repeat of 'binary'        

#------Prepare to start Routine "instr_infer_intro"-------
t = 0
instr_infer_introClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instr_infer_intro_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
instr_infer_intro_resp.status = NOT_STARTED
# keep track of which components have finished
instr_infer_introComponents = []
instr_infer_introComponents.append(instr_infer_intro_txt)
instr_infer_introComponents.append(instr_infer_intro_resp)
for thisComponent in instr_infer_introComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr_infer_intro"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instr_infer_introClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_infer_intro_txt* updates
    if t >= 0.0 and instr_infer_intro_txt.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_infer_intro_txt.tStart = t  # underestimates by a little under one frame
        instr_infer_intro_txt.frameNStart = frameN  # exact frame index
        instr_infer_intro_txt.setAutoDraw(True)
    
    # *instr_infer_intro_resp* updates
    if t >= 5.0 and instr_infer_intro_resp.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_infer_intro_resp.tStart = t  # underestimates by a little under one frame
        instr_infer_intro_resp.frameNStart = frameN  # exact frame index
        instr_infer_intro_resp.status = STARTED
        # keyboard checking is just starting
        instr_infer_intro_resp.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if instr_infer_intro_resp.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            instr_infer_intro_resp.keys = theseKeys[-1]  # just the last key pressed
            instr_infer_intro_resp.rt = instr_infer_intro_resp.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_infer_introComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr_infer_intro"-------
for thisComponent in instr_infer_introComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instr_infer_intro_resp.keys in ['', [], None]:  # No response was made
   instr_infer_intro_resp.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('instr_infer_intro_resp.keys',instr_infer_intro_resp.keys)
if instr_infer_intro_resp.keys != None:  # we had a response
    thisExp.addData('instr_infer_intro_resp.rt', instr_infer_intro_resp.rt)
thisExp.nextEntry()


#------Prepare to start Routine "instr_infer_practice"-------
t = 0
instr_infer_practiceClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instr_infer_practice_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
instr_infer_practice_resp.status = NOT_STARTED
# keep track of which components have finished
instr_infer_practiceComponents = []
instr_infer_practiceComponents.append(instr_infer_practice_txt)
instr_infer_practiceComponents.append(instr_infer_practice_resp)
for thisComponent in instr_infer_practiceComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr_infer_practice"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instr_infer_practiceClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_infer_practice_txt* updates
    if t >= 0.0 and instr_infer_practice_txt.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_infer_practice_txt.tStart = t  # underestimates by a little under one frame
        instr_infer_practice_txt.frameNStart = frameN  # exact frame index
        instr_infer_practice_txt.setAutoDraw(True)
    
    # *instr_infer_practice_resp* updates
    if t >= 5.0 and instr_infer_practice_resp.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_infer_practice_resp.tStart = t  # underestimates by a little under one frame
        instr_infer_practice_resp.frameNStart = frameN  # exact frame index
        instr_infer_practice_resp.status = STARTED
        # keyboard checking is just starting
        instr_infer_practice_resp.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if instr_infer_practice_resp.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            instr_infer_practice_resp.keys = theseKeys[-1]  # just the last key pressed
            instr_infer_practice_resp.rt = instr_infer_practice_resp.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_infer_practiceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr_infer_practice"-------
for thisComponent in instr_infer_practiceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instr_infer_practice_resp.keys in ['', [], None]:  # No response was made
   instr_infer_practice_resp.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('instr_infer_practice_resp.keys',instr_infer_practice_resp.keys)
if instr_infer_practice_resp.keys != None:  # we had a response
    thisExp.addData('instr_infer_practice_resp.rt', instr_infer_practice_resp.rt)
thisExp.nextEntry()


######################## PRACTICE LOOP ##########################

# set up handler to look after randomisation of conditions etc
practice_loop = data.TrialHandler(nReps=1, method=u'sequential', 
    extraInfo=expInfo, originPath=None,
    trialList=data.importConditions(filename+'_practicecond.csv'),
    seed=None, name='practice_loop')
thisExp.addLoop(practice_loop)  # add the loop to the experiment
thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisPractice_loop.rgb)
if thisPractice_loop != None:
    for paramName in thisPractice_loop.keys():
        exec(paramName + '= thisPractice_loop.' + paramName)

  
for thisPractice_loop in practice_loop:
    currentLoop = practice_loop
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop.keys():
            exec(paramName + '= thisPractice_loop.' + paramName)    
    
        
    # Define class for setting the feedback image
    class set_feedback:
        options = ['correct', 'wrong']
        feedback = np.random.choice(options, replace=True, p=[0.8, 0.2])
        if feedback=='correct':
            feedback_img = img_correct
        elif feedback=='wrong':
            feedback_img = img_wrong

        if feedback_img==img_left:
            feedback_side = 'left'
        elif feedback_img==img_right:
            feedback_side = 'right'


    #------Prepare to start Routine "trial"-------
    t = 0
    trialtime = datetime.datetime.now()
    trialClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    img_left_infer.setImage(dir_path+img_left)
    img_right_infer.setImage(dir_path+img_right)
    infer_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
    infer_resp.status = NOT_STARTED
    # keep track of which components have finished
    trialComponents = []
    trialComponents.append(fixation_text)
    trialComponents.append(img_left_infer)
    trialComponents.append(img_right_infer)
    trialComponents.append(infer_resp)
    for thisComponent in trialComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "trial"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_text* updates
        if t >= 0 and fixation_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            fixation_text.tStart = t  # underestimates by a little under one frame
            fixation_text.frameNStart = frameN  # exact frame index
            fixation_text.setAutoDraw(True)
        elif fixation_text.status == STARTED and t>= (0 + (2.0-win.monitorFramePeriod*0.75)):
            fixation_text.setAutoDraw(False)

        # *img_left_infer* updates
        if t >= 2.0 and img_left_infer.status == NOT_STARTED:
            # keep track of start time/frame for later
            img_left_infer.tStart = t  # underestimates by a little under one frame
            img_left_infer.frameNStart = frameN  # exact frame index
            img_left_infer.setAutoDraw(True)
        
        # *img_right_infer* updates
        if t >= 2.0 and img_right_infer.status == NOT_STARTED:
            # keep track of start time/frame for later
            img_right_infer.tStart = t  # underestimates by a little under one frame
            img_right_infer.frameNStart = frameN  # exact frame index
            img_right_infer.setAutoDraw(True)
                  
        # *infer_resp* updates
        if t >= 2.0 and infer_resp.status == NOT_STARTED:
            # keep track of start time/frame for later
            infer_resp.tStart = t  # underestimates by a little under one frame
            infer_resp.frameNStart = frameN  # exact frame index
            infer_resp.status = STARTED
            # keyboard checking is just starting
            infer_resp.clock.reset()  # now t=0
            event.clearEvents(eventType='keyboard')
        if infer_resp.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                infer_resp.keys = theseKeys[-1]  # just the last key pressed
                infer_resp.rt = infer_resp.clock.getTime()
                # a response ends the routine
                continueRoutine = False

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()
    
    #-------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if infer_resp.keys in ['', [], None]:  # No response was made
       infer_resp.keys=None
    # # store data for practice_loop (TrialHandler)
    # practice_loop.addData('infer_resp.keys',infer_resp.keys)
    # if infer_resp.keys != None:  # we had a response
    #     practice_loop.addData('infer_resp.rt', infer_resp.rt)
    practice_loop.addData('set_feedback.feedback_img', set_feedback.feedback_img)
    practice_loop.addData('trial_start_time', trialtime)
    
    # # Increase correct and incorrect response counters by 1 based on S's response
    # if infer_resp.keys=='left' and img_correct==img_left:
    #     correct_counter += 1
    # elif infer_resp.keys=='left' and img_correct==img_right:
    #     incorrect_counter += 1
    # elif infer_resp.keys=='right' and img_correct==img_right:
    #     correct_counter += 1
    # elif infer_resp.keys=='right' and img_correct==img_left:
    #     incorrect_counter += 1
    

    #------Prepare to start Routine "trial_post_response"-------
    t = 0
    trial_post_responseClock.reset()  # clock 
    frameN = -1
    routineTimer.add(3.000000)
    # keep track of which components have finished
    trial_post_responseComponents = []
    trial_post_responseComponents.append(img_left_infer)
    trial_post_responseComponents.append(img_right_infer)
    trial_post_responseComponents.append(selection_arrow_left)
    trial_post_responseComponents.append(selection_arrow_right)
    trial_post_responseComponents.append(feedback_box_left)
    trial_post_responseComponents.append(feedback_box_right)
    for thisComponent in trial_post_responseComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "trial_post_response"-------
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trial_post_responseClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *img_left_infer* updates
        if t >= 0.0 and img_left_infer.status == NOT_STARTED:
            # keep track of start time/frame for later
            img_left_infer.tStart = t  # underestimates by a little under one frame
            img_left_infer.frameNStart = frameN  # exact frame index
            img_left_infer.setAutoDraw(True)
        elif img_left_infer.status == STARTED and t >= (0.0 + (3.0-win.monitorFramePeriod*0.75)): #most of one frame period left
            img_left_infer.setAutoDraw(False)
        
        # *img_right_infer* updates
        if t >= 0.0 and img_right_infer.status == NOT_STARTED:
            # keep track of start time/frame for later
            img_right_infer.tStart = t  # underestimates by a little under one frame
            img_right_infer.frameNStart = frameN  # exact frame index
            img_right_infer.setAutoDraw(True)
        elif img_right_infer.status == STARTED and t >= (0.0 + (3.0-win.monitorFramePeriod*0.75)): #most of one frame period left
            img_right_infer.setAutoDraw(False)
        
        if infer_resp.keys == 'left':
            # *selection_arrow_left* updates
            if t >= 0.0 and selection_arrow_left.status == NOT_STARTED:
                # keep track of start time/frame for later
                selection_arrow_left.tStart = t  # underestimates by a little under one frame
                selection_arrow_left.frameNStart = frameN  # exact frame index
                selection_arrow_left.setAutoDraw(True)
            elif selection_arrow_left.status == STARTED and t >= (0.0 + (0.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                selection_arrow_left.setAutoDraw(False)
        
        if infer_resp.keys == 'right':
            # *selection_arrow_right* updates
            if t >= 0.0 and selection_arrow_right.status == NOT_STARTED:
                # keep track of start time/frame for later
                selection_arrow_right.tStart = t  # underestimates by a little under one frame
                selection_arrow_right.frameNStart = frameN  # exact frame index
                selection_arrow_right.setAutoDraw(True)
            elif selection_arrow_right.status == STARTED and t >= (0.0 + (0.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                selection_arrow_right.setAutoDraw(False)
        
        if set_feedback.feedback_side == 'left':
            # *feedback_box_left* updates
            if t >= 1.0 and feedback_box_left.status == NOT_STARTED:
                # keep track of start time/frame for later
                feedback_box_left.tStart = t  # underestimates by a little under one frame
                feedback_box_left.frameNStart = frameN  # exact frame index
                feedback_box_left.setAutoDraw(True)
            elif feedback_box_left.status == STARTED and t >= (1.0 + (2.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                feedback_box_left.setAutoDraw(False)
        
        if set_feedback.feedback_side == 'right':
            # *feedback_box_right* updates
            if t >= 1.0 and feedback_box_right.status == NOT_STARTED:
                # keep track of start time/frame for later
                feedback_box_right.tStart = t  # underestimates by a little under one frame
                feedback_box_right.frameNStart = frameN  # exact frame index
                feedback_box_right.setAutoDraw(True)
            elif feedback_box_right.status == STARTED and t >= (1.0 + (2.0-win.monitorFramePeriod*0.75)): #most of one frame period right
                feedback_box_right.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_post_responseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "trial_post_response"-------
    for thisComponent in trial_post_responseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    
    thisExp.nextEntry()
                        
# completed 1 repeat of 'practice_loop'


#------Prepare to start Routine "instr_infer"-------
t = 0
instr_inferClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instr_infer_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
instr_infer_resp.status = NOT_STARTED
# keep track of which components have finished
instr_inferComponents = []
instr_inferComponents.append(instr_infer_txt)
instr_inferComponents.append(instr_infer_resp)
for thisComponent in instr_inferComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr_infer"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instr_inferClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instr_infer_txt* updates
    if t >= 0.0 and instr_infer_txt.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_infer_txt.tStart = t  # underestimates by a little under one frame
        instr_infer_txt.frameNStart = frameN  # exact frame index
        instr_infer_txt.setAutoDraw(True)
    
    # *instr_infer_resp* updates
    if t >= 5.0 and instr_infer_resp.status == NOT_STARTED:
        # keep track of start time/frame for later
        instr_infer_resp.tStart = t  # underestimates by a little under one frame
        instr_infer_resp.frameNStart = frameN  # exact frame index
        instr_infer_resp.status = STARTED
        # keyboard checking is just starting
        instr_infer_resp.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if instr_infer_resp.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            instr_infer_resp.keys = theseKeys[-1]  # just the last key pressed
            instr_infer_resp.rt = instr_infer_resp.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_inferComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr_infer"-------
for thisComponent in instr_inferComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instr_infer_resp.keys in ['', [], None]:  # No response was made
   instr_infer_resp.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('instr_infer_resp.keys',instr_infer_resp.keys)
if instr_infer_resp.keys != None:  # we had a response
    thisExp.addData('instr_infer_resp.rt', instr_infer_resp.rt)
thisExp.nextEntry()


######################## BLOCK LOOP ##########################

# This loop runs a second nested loop that shows each item pair 10 times, followed by a rest break prompt.

# set up handler to look after randomisation of conditions etc
block_loop = data.TrialHandler(nReps=3, method=u'sequential', 
    extraInfo=expInfo, originPath=None,
    trialList=[None],
    seed=None, name='block_loop')
thisExp.addLoop(block_loop)  # add the loop to the experiment
thisBlock_loop = block_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisBlock_loop.rgb)
if thisBlock_loop != None:
    for paramName in thisBlock_loop.keys():
        exec(paramName + '= thisBlock_loop.' + paramName)

# Set up counter to keep track of block loop cycle, so that the rest prompt is only shown twice
block_loop_counter = 0


for thisBlock_loop in block_loop:
    currentLoop = block_loop
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
    if thisBlock_loop != None:
        for paramName in thisBlock_loop.keys():
            exec(paramName + '= thisBlock_loop.' + paramName)
    
    # Increase loop counter by one
    block_loop_counter += 1
    
    # Change the condition file for the feedback block based on which block loop we're on
    condfilename = filename+u'_block'+str(block_loop_counter)+u'cond.csv'
    
    # # Minimize the psychopy window so the calibration window can be seen
    # win.winHandle.minimize()
    # #Do the eye tracker setup at the beginning of each block
    # tracker.runSetupProcedure()
    # # Re-display the psychopy window after setup is completed
    # win.winHandle.maximize()
    # win.winHandle.activate()
    

    
    #------Prepare to start Routine "get_ready"-------
    t = 0
    get_readyClock.reset()  # clock 
    frameN = -1
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    get_ready_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
    get_ready_resp.status = NOT_STARTED
    # keep track of which components have finished
    get_readyComponents = []
    get_readyComponents.append(get_ready_text)
    get_readyComponents.append(get_ready_resp)
    for thisComponent in get_readyComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    
    #-------Start Routine "get_ready"-------
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = get_readyClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *get_ready_text* updates
        if t >= 0.0 and get_ready_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            get_ready_text.tStart = t  # underestimates by a little under one frame
            get_ready_text.frameNStart = frameN  # exact frame index
            get_ready_text.setAutoDraw(True)
        
        # *get_ready_resp* updates
        if t >= 0.0 and get_ready_resp.status == NOT_STARTED:
            # keep track of start time/frame for later
            get_ready_resp.tStart = t  # underestimates by a little under one frame
            get_ready_resp.frameNStart = frameN  # exact frame index
            get_ready_resp.status = STARTED
            # keyboard checking is just starting
            get_ready_resp.clock.reset()  # now t=0
            event.clearEvents(eventType='keyboard')
        if get_ready_resp.status == STARTED:
            theseKeys = event.getKeys(keyList=['p'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                get_ready_resp.keys = theseKeys[-1]  # just the last key pressed
                get_ready_resp.rt = get_ready_resp.clock.getTime()
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in get_readyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()

    #-------Ending Routine "get_ready"-------
    for thisComponent in get_readyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if get_ready_resp.keys in ['', [], None]:  # No response was made
       get_ready_resp.keys=None
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('get_ready_resp.keys',get_ready_resp.keys)
    if get_ready_resp.keys != None:  # we had a response
        thisExp.addData('get_ready_resp.rt', get_ready_resp.rt)
    thisExp.nextEntry()

    ######################## TRIAL LOOP ##########################

    # set up handler to look after randomisation of conditions etc
    trial_loop = data.TrialHandler(nReps=1, method=u'sequential', 
        extraInfo=expInfo, originPath=None,
        trialList=data.importConditions(condfilename),
        seed=None, name='trial_loop')
    thisExp.addLoop(trial_loop)  # add the loop to the experiment
    thisTrial_loop = trial_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb=thisTrial_loop.rgb)
    if thisTrial_loop != None:
        for paramName in thisTrial_loop.keys():
            exec(paramName + '= thisTrial_loop.' + paramName)
    
      
    for thisTrial_loop in trial_loop:
        currentLoop = trial_loop
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop.keys():
                exec(paramName + '= thisTrial_loop.' + paramName)    
        
            
        # Define class for setting the feedback image
        class set_feedback:
            options = ['correct', 'wrong']
            feedback = np.random.choice(options, replace=True, p=[0.8, 0.2])
            if feedback=='correct':
                feedback_img = img_correct
            elif feedback=='wrong':
                feedback_img = img_wrong

            if feedback_img==img_left:
                feedback_side = 'left'
            elif feedback_img==img_right:
                feedback_side = 'right'
                        
        # # Start getting data from the eye tracker
        # tracker.enableEventReporting(True)
        # # Send beginning-of-trial messages to eye tracker data file
        # trial_number = (block_loop.thisN * 200) + trial_loop.thisTrialN # Trial number out of 599 (starts at 0)
        # tracker.sendCommand("record_status_message 'INFERRING, Block %d/3, Trial %d/600 '" % (block_loop.thisN + 1, trial_number + 1))
        # tracker.sendMessage("TRIALID %d" % trial_number)
        # tracker.sendMessage("!V TRIAL_VAR_DATA %d" % trial_number)

        
        #------Prepare to start Routine "trial"-------
        t = 0
        trialtime = datetime.datetime.now()
        trialClock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        img_left_infer.setImage(dir_path+img_left)
        img_right_infer.setImage(dir_path+img_right)
        infer_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
        infer_resp.status = NOT_STARTED
        # keep track of which components have finished
        trialComponents = []
        trialComponents.append(fixation_text)
        trialComponents.append(img_left_infer)
        trialComponents.append(img_right_infer)
        trialComponents.append(infer_resp)
        for thisComponent in trialComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # Set indicator variable for whether eye tracker messages were sent during the first loop of the routine
        eye_sync_messages_sent = 0
        
        #-------Start Routine "trial"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = trialClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_text* updates
            if t >= 0 and fixation_text.status == NOT_STARTED:
                if eye_sync_messages_sent==0:
                    startTime = pylink.currentTime() # Get time at start of stimulus drawing
                # keep track of start time/frame for later
                fixation_text.tStart = t  # underestimates by a little under one frame
                fixation_text.frameNStart = frameN  # exact frame index
                fixation_text.setAutoDraw(True)
                if eye_sync_messages_sent==0:
                    drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                    # tracker.sendMessage("%d DISPLAY ON" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                    # tracker.sendMessage("SYNCTIME %d" %drawTime)
                    eye_sync_messages_sent = 1
            elif fixation_text.status == STARTED and t>= (0 + (2.0-win.monitorFramePeriod*0.75)):
                fixation_text.setAutoDraw(False)
            
            # *img_left_infer* updates
            if t >= 2.0 and img_left_infer.status == NOT_STARTED:
                if eye_sync_messages_sent==1:
                    startTime = pylink.currentTime() # Get time at start of stimulus drawing
                # keep track of start time/frame for later
                img_left_infer.tStart = t  # underestimates by a little under one frame
                img_left_infer.frameNStart = frameN  # exact frame index
                img_left_infer.setAutoDraw(True)
            
            # *img_right_infer* updates
            if t >= 2.0 and img_right_infer.status == NOT_STARTED:
                # keep track of start time/frame for later
                img_right_infer.tStart = t  # underestimates by a little under one frame
                img_right_infer.frameNStart = frameN  # exact frame index
                img_right_infer.setAutoDraw(True)
                if eye_sync_messages_sent==1:
                    drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                    # tracker.sendMessage("%d DISPLAY ON ITEMS" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                    eye_sync_messages_sent = 2
                      
            # *infer_resp* updates
            if t >= 2.0 and infer_resp.status == NOT_STARTED:
                # keep track of start time/frame for later
                infer_resp.tStart = t  # underestimates by a little under one frame
                infer_resp.frameNStart = frameN  # exact frame index
                infer_resp.status = STARTED
                # keyboard checking is just starting
                infer_resp.clock.reset()  # now t=0
                event.clearEvents(eventType='keyboard')
            if infer_resp.status == STARTED:
                theseKeys = event.getKeys(keyList=['left', 'right'])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    infer_resp.keys = theseKeys[-1]  # just the last key pressed
                    infer_resp.rt = infer_resp.clock.getTime()
                    # a response ends the routine
                    continueRoutine = False
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineTimer.reset()  # if we abort early the non-slip timer needs reset
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                # tracker.enableEventReporting(False) # End eye tracker data recording
                # tracker.sendMessage("EXPERIMENT ABORTED")
                # io.quit() # Close iohub
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
            else:  # this Routine was not non-slip safe so reset non-slip timer
                routineTimer.reset()
        
        #-------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if infer_resp.keys in ['', [], None]:  # No response was made
           infer_resp.keys=None
        # store data for trial_loop (TrialHandler)
        trial_loop.addData('infer_resp.keys',infer_resp.keys)
        if infer_resp.keys != None:  # we had a response
            trial_loop.addData('infer_resp.rt', infer_resp.rt)
        trial_loop.addData('set_feedback.feedback_img', set_feedback.feedback_img)
        trial_loop.addData('trial_start_time', trialtime)
        
        # Increase correct and incorrect response counters by 1 based on S's response
        if infer_resp.keys=='left' and img_correct==img_left:
            correct_counter += 1
        elif infer_resp.keys=='left' and img_correct==img_right:
            incorrect_counter += 1
        elif infer_resp.keys=='right' and img_correct==img_right:
            correct_counter += 1
        elif infer_resp.keys=='right' and img_correct==img_left:
            incorrect_counter += 1
        

        #------Prepare to start Routine "trial_post_response"-------
        t = 0
        trial_post_responseClock.reset()  # clock 
        frameN = -1
        routineTimer.add(3.000000)
        # keep track of which components have finished
        trial_post_responseComponents = []
        trial_post_responseComponents.append(img_left_infer)
        trial_post_responseComponents.append(img_right_infer)
        trial_post_responseComponents.append(selection_arrow_left)
        trial_post_responseComponents.append(selection_arrow_right)
        trial_post_responseComponents.append(feedback_box_left)
        trial_post_responseComponents.append(feedback_box_right)
        for thisComponent in trial_post_responseComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # Set indicator variable for whether eye tracker messages were sent during the first loop of the routine
        eye_sync_messages_sent = 0

        #-------Start Routine "trial_post_response"-------
        continueRoutine = True
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trial_post_responseClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame

            # *img_left_infer* updates
            if t >= 0.0 and img_left_infer.status == NOT_STARTED:
                # keep track of start time/frame for later
                img_left_infer.tStart = t  # underestimates by a little under one frame
                img_left_infer.frameNStart = frameN  # exact frame index
                img_left_infer.setAutoDraw(True)
            elif img_left_infer.status == STARTED and t >= (0.0 + (3.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                img_left_infer.setAutoDraw(False)
            
            # *img_right_infer* updates
            if t >= 0.0 and img_right_infer.status == NOT_STARTED:
                # keep track of start time/frame for later
                img_right_infer.tStart = t  # underestimates by a little under one frame
                img_right_infer.frameNStart = frameN  # exact frame index
                img_right_infer.setAutoDraw(True)
            elif img_right_infer.status == STARTED and t >= (0.0 + (3.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                img_right_infer.setAutoDraw(False)
            
            if infer_resp.keys == 'left':
                # *selection_arrow_left* updates
                if t >= 0.0 and selection_arrow_left.status == NOT_STARTED:
                    if eye_sync_messages_sent==0:
                        startTime = pylink.currentTime() # Get time at start of stimulus drawing
                    # keep track of start time/frame for later
                    selection_arrow_left.tStart = t  # underestimates by a little under one frame
                    selection_arrow_left.frameNStart = frameN  # exact frame index
                    selection_arrow_left.setAutoDraw(True)
                    if eye_sync_messages_sent==0:
                        drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                        # tracker.sendMessage("%d DISPLAY ON SELECTION" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                        eye_sync_messages_sent = 1
                elif selection_arrow_left.status == STARTED and t >= (0.0 + (0.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    if eye_sync_messages_sent==1:
                        startTime = pylink.currentTime() # Get time at start of stimulus drawing
                    selection_arrow_left.setAutoDraw(False)
                    if eye_sync_messages_sent==1:
                        drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                        # tracker.sendMessage("%d DISPLAY ON SELECTION OFF" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                        eye_sync_messages_sent = 2
            
            if infer_resp.keys == 'right':
                # *selection_arrow_right* updates
                if t >= 0.0 and selection_arrow_right.status == NOT_STARTED:
                    if eye_sync_messages_sent==0:
                        startTime = pylink.currentTime() # Get time at start of stimulus drawing
                    # keep track of start time/frame for later
                    selection_arrow_right.tStart = t  # underestimates by a little under one frame
                    selection_arrow_right.frameNStart = frameN  # exact frame index
                    selection_arrow_right.setAutoDraw(True)
                    if eye_sync_messages_sent==0:
                        drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                        # tracker.sendMessage("%d DISPLAY ON SELECTION" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                        eye_sync_messages_sent = 1
                elif selection_arrow_right.status == STARTED and t >= (0.0 + (0.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    if eye_sync_messages_sent==1:
                        startTime = pylink.currentTime() # Get time at start of stimulus drawing
                    selection_arrow_right.setAutoDraw(False)
                    if eye_sync_messages_sent==1:
                        drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                        # tracker.sendMessage("%d DISPLAY ON SELECTION OFF" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                        eye_sync_messages_sent = 2
            
            if set_feedback.feedback_side == 'left':
                # *feedback_box_left* updates
                if t >= 1.0 and feedback_box_left.status == NOT_STARTED:
                    if eye_sync_messages_sent==2:
                        startTime = pylink.currentTime() # Get time at start of stimulus drawing
                    # keep track of start time/frame for later
                    feedback_box_left.tStart = t  # underestimates by a little under one frame
                    feedback_box_left.frameNStart = frameN  # exact frame index
                    feedback_box_left.setAutoDraw(True)
                    if eye_sync_messages_sent==2:
                        drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                        # tracker.sendMessage("%d DISPLAY ON FEEDBACK" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                        eye_sync_messages_sent = 3
                elif feedback_box_left.status == STARTED and t >= (1.0 + (2.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                    feedback_box_left.setAutoDraw(False)
            
            if set_feedback.feedback_side == 'right':
                # *feedback_box_right* updates
                if t >= 1.0 and feedback_box_right.status == NOT_STARTED:
                    if eye_sync_messages_sent==2:
                        startTime = pylink.currentTime() # Get time at start of stimulus drawing
                    # keep track of start time/frame for later
                    feedback_box_right.tStart = t  # underestimates by a little under one frame
                    feedback_box_right.frameNStart = frameN  # exact frame index
                    feedback_box_right.setAutoDraw(True)
                    if eye_sync_messages_sent==2:
                        drawTime = (pylink.currentTime() - startTime) # Calculate time it took to draw both items
                        # tracker.sendMessage("%d DISPLAY ON FEEDBACK" %drawTime) # Send messages to eye tracker data file to mark stimulus onset
                        eye_sync_messages_sent = 3
                elif feedback_box_right.status == STARTED and t >= (1.0 + (2.0-win.monitorFramePeriod*0.75)): #most of one frame period right
                    feedback_box_right.setAutoDraw(False)


            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineTimer.reset()  # if we abort early the non-slip timer needs reset
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_post_responseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                # tracker.enableEventReporting(False) # End eye tracker data recording
                # tracker.sendMessage("EXPERIMENT ABORTED")
                # io.quit() # Close iohub
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "trial_post_response"-------
        for thisComponent in trial_post_responseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)



        thisExp.nextEntry()
                            
    # completed 5 repeats of 'trial_loop'
    
    if block_loop_counter <= 2: # If this is the first or second cycle of the loop, display the rest break prompt

        #------Prepare to start Routine "rest_prompt"-------
        t = 0
        rest_promptClock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        rest_prompt_resp = event.BuilderKeyResponse()  # create an object of type KeyResponse
        rest_prompt_resp.status = NOT_STARTED
        # keep track of which components have finished
        rest_promptComponents = []
        rest_promptComponents.append(rest_prompt_txt)
        rest_promptComponents.append(rest_prompt_resp)
        for thisComponent in rest_promptComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        #-------Start Routine "rest_prompt"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = rest_promptClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_prompt_txt* updates
            if t >= 0.0 and rest_prompt_txt.status == NOT_STARTED:
                # keep track of start time/frame for later
                rest_prompt_txt.tStart = t  # underestimates by a little under one frame
                rest_prompt_txt.frameNStart = frameN  # exact frame index
                rest_prompt_txt.setAutoDraw(True)
            
            # *rest_prompt_resp* updates
            if t >= 2.0 and rest_prompt_resp.status == NOT_STARTED:
                # keep track of start time/frame for later
                rest_prompt_resp.tStart = t  # underestimates by a little under one frame
                rest_prompt_resp.frameNStart = frameN  # exact frame index
                rest_prompt_resp.status = STARTED
                # keyboard checking is just starting
                rest_prompt_resp.clock.reset()  # now t=0
                event.clearEvents(eventType='keyboard')
            if rest_prompt_resp.status == STARTED:
                theseKeys = event.getKeys(keyList=['space'])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    rest_prompt_resp.keys = theseKeys[-1]  # just the last key pressed
                    rest_prompt_resp.rt = rest_prompt_resp.clock.getTime()
                    # a response ends the routine
                    continueRoutine = False
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineTimer.reset()  # if we abort early the non-slip timer needs reset
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest_promptComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                # tracker.enableEventReporting(False) # End eye tracker data recording
                # tracker.sendMessage("EXPERIMENT ABORTED")
                # io.quit() # CLose iohub
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
            else:  # this Routine was not non-slip safe so reset non-slip timer
                routineTimer.reset()

        #-------Ending Routine "rest_prompt"-------
        for thisComponent in rest_promptComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if rest_prompt_resp.keys in ['', [], None]:  # No response was made
           rest_prompt_resp.keys=None
        # store data for thisExp (ExperimentHandler)
        thisExp.addData('rest_prompt_resp.keys',rest_prompt_resp.keys)
        if rest_prompt_resp.keys != None:  # we had a response
            thisExp.addData('rest_prompt_resp.rt', rest_prompt_resp.rt)


    thisExp.nextEntry()
    
# completed 3 repeats of 'block_loop'


# store data for total number of correct and incorrect responses    
thisExp.addData('correct_counter', correct_counter)
thisExp.addData('incorrect_counter', incorrect_counter)
thisExp.nextEntry()     

# tracker.setConnectionState(False) # Close and transfer eye-tracking data, then close down eye tracker connection


######################## AUCTION ##########################

if len(prefs)==41 and len(bids)==41: # If S completed the entire binary choice and BDM routines, run the auction

    # run the auction
    class auction:
        rand_itm = randint(1, (len(prefs))) # pick index for S's chosen item from a random choice
        price = float((randint(1, 300)))/100 # assign a price to the item randomly from between 0.01 and 3 pounds
        bid = prefs[rand_itm][3]
        # check if S's bid for that item was above or below the price
        if bid >= price:
            win_item=True
        elif bid < price:
            win_item=False

        # set text for the auction screen
        if win_item==True:
            auc_res_txt = u'Congratulations! You won the following item at auction. \nThis was your preferred item out of a randomly selected pair from one of the choice tasks.'
            auc_prc_txt = u'Your bid of \xa3' + '{0:.2f}'.format(bid) + u' matched or exceeded the randomly generated price of \xa3' + '{0:.2f}'.format(price) + '.\n\n[Press space bar to continue]'
            cost = price
        elif win_item==False:
            auc_res_txt = u'Sorry, you did not win the following item at auction. \nThis was your preferred item out of a randomly selected pair from one of the choice tasks.'
            auc_prc_txt = u'Your bid of \xa3' + '{0:.2f}'.format(bid) + u' was lower than the randomly generated price of \xa3' + '{0:.2f}'.format(price) + '.\n\n[Press space bar to continue]'
            cost = 0

        # find the image path of the auction item
        rand_itm_img = prefs[rand_itm][2]

    # add up each component of S's payment
    class score:
        base = 25
        rwrd = 0.01
        loss = 0
        rwrd_tot = rwrd * correct_counter
        loss_tot = loss * incorrect_counter
        pre_pymt = base + rwrd_tot - loss_tot
        final_pymt = pre_pymt - auction.cost

        pymt_expl_bdwn = u'\xa3' + '{0:.2f}'.format(base) + u' base payment'+ u'\n+ \xa3' + '{0:.2f}'.format(rwrd_tot) + ' reward for ' + str(correct_counter) + u' correct responses (\xa3' + '{0:.2f}'.format(rwrd) + u' each)' + u'\n- \xa3' + '{0:.2f}'.format(auction.cost) + ' cost of auction item' 
        pymt_expl_tot = u'Your total payment is: \xa3' + '{0:.2f}'.format(final_pymt) + '\n\nThank you for participating!'


    # Initialize components for Routine "auc_disp"
    auc_dispClock = core.Clock()
    pic_auc_itm = visual.ImageStim(win=win, name='pic_auc_itm',
        image=dir_path+auction.rand_itm_img, mask=None,
        ori=0, pos=[0, 0], size=None,
        color=[1,1,1], colorSpace=u'rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=False, depth=0.0)
    auc_txt1 = visual.TextStim(win=win, ori=0, name='auc_txt1',
        text=auction.auc_res_txt, font=u'Arial',
        pos=[0, 0.6], height=0.07, wrapWidth=None,
        color=u'white', colorSpace=u'rgb', opacity=1,
        depth=0.0)
    auc_txt2 = visual.TextStim(win=win, ori=0, name='auc_txt2',
        text=auction.auc_prc_txt, font=u'Arial',
        pos=[0, -0.6], height=0.07, wrapWidth=None,
        color=u'white', colorSpace=u'rgb', opacity=1,
        depth=0.0)


    #------Prepare to start Routine "auc_disp"-------
    t = 0
    auc_dispClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    key_resp_auc_disp = event.BuilderKeyResponse()  # create an object of type KeyResponse
    key_resp_auc_disp.status = NOT_STARTED
    # keep track of which components have finished
    auc_dispComponents = []
    auc_dispComponents.append(pic_auc_itm)
    auc_dispComponents.append(auc_txt1)
    auc_dispComponents.append(auc_txt2)
    for thisComponent in auc_dispComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "auc_disp"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = auc_dispClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pic_auc_itm* updates
        if t >= 0.2 and pic_auc_itm.status == NOT_STARTED:
            # keep track of start time/frame for later
            pic_auc_itm.tStart = t  # underestimates by a little under one frame
            pic_auc_itm.frameNStart = frameN  # exact frame index
            pic_auc_itm.setAutoDraw(True)
        
        # *auc_txt1* updates
        if t >= 0.2 and auc_txt1.status == NOT_STARTED:
            # keep track of start time/frame for later
            auc_txt1.tStart = t  # underestimates by a little under one frame
            auc_txt1.frameNStart = frameN  # exact frame index
            auc_txt1.setAutoDraw(True)

        # *auc_txt2* updates
        if t >= 0.2 and auc_txt2.status == NOT_STARTED:
            # keep track of start time/frame for later
            auc_txt2.tStart = t  # underestimates by a little under one frame
            auc_txt2.frameNStart = frameN  # exact frame index
            auc_txt2.setAutoDraw(True)
        
        # *key_resp_auc_disp* updates
        if t >= 0.2 and key_resp_auc_disp.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_resp_auc_disp.tStart = t  # underestimates by a little under one frame
            key_resp_auc_disp.frameNStart = frameN  # exact frame index
            key_resp_auc_disp.status = STARTED
            # keyboard checking is just starting
            key_resp_auc_disp.clock.reset()  # now t=0
        if key_resp_auc_disp.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                key_resp_auc_disp.keys = theseKeys[-1]  # just the last key pressed
                key_resp_auc_disp.rt = key_resp_auc_disp.clock.getTime()
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in auc_dispComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()

    #-------Ending Routine "auc_disp"-------
    for thisComponent in auc_dispComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_resp_auc_disp.keys in ['', [], None]:  # No response was made
       key_resp_auc_disp.keys=None
    # store data for thisExp (TrialHandler)
    thisExp.addData('auction.win_item', auction.win_item)
    thisExp.addData('auction.price', auction.price)
    thisExp.addData('auction.bid', auction.bid)
    thisExp.addData('auction.rand_itm_img', auction.rand_itm_img)
    thisExp.addData('score.base', score.base)
    thisExp.addData('score.rwrd', score.rwrd)
    thisExp.addData('score.loss', score.loss)
    thisExp.addData('correct_counter', correct_counter)
    thisExp.addData('incorrect_counter', incorrect_counter)
    thisExp.addData('score.final_pymt', score.final_pymt)
    thisExp.nextEntry()


    # Initialize components for Routine "pymt_disp"
    pymt_dispClock = core.Clock()
    pymt_disp_txt1 = visual.TextStim(win=win, ori=0, name='pymt_disp_txt1',
        text=score.pymt_expl_bdwn, font=u'Arial',
        pos=[0, 0.6], height=0.07, wrapWidth=1.5,
        color=u'white', colorSpace=u'rgb', opacity=1,
        depth=0.0)
    pymt_disp_txt2 = visual.TextStim(win=win, ori=0, name='pymt_disp_txt2',
        text=score.pymt_expl_tot, font=u'Arial',
        pos=[0, -0.65], height=0.1, wrapWidth=None,
        color=u'white', colorSpace=u'rgb', opacity=1,
        depth=0.0)

    #------Prepare to start Routine "pymt_disp"-------
    t = 0
    pymt_dispClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    key_resp_pymt_disp = event.BuilderKeyResponse()  # create an object of type KeyResponse
    key_resp_pymt_disp.status = NOT_STARTED
    # keep track of which components have finished
    pymt_dispComponents = []
    pymt_dispComponents.append(pic_auc_itm)
    pymt_dispComponents.append(pymt_disp_txt1)
    pymt_dispComponents.append(pymt_disp_txt2)
    pymt_dispComponents.append(key_resp_pymt_disp)
    for thisComponent in pymt_dispComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "pymt_disp"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = pymt_dispClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        if auction.win_item==True:
            # *pic_auc_itm* updates
            if t >= 0.0 and pic_auc_itm.status == NOT_STARTED:
                # keep track of start time/frame for later
                pic_auc_itm.tStart = t  # underestimates by a little under one frame
                pic_auc_itm.frameNStart = frameN  # exact frame index
                pic_auc_itm.setAutoDraw(True)

        # *pymt_disp_txt1* updates
        if t >= 0.0 and pymt_disp_txt1.status == NOT_STARTED:
            # keep track of start time/frame for later
            pymt_disp_txt1.tStart = t  # underestimates by a little under one frame
            pymt_disp_txt1.frameNStart = frameN  # exact frame index
            pymt_disp_txt1.setAutoDraw(True)

        # *pymt_disp_txt2* updates
        if t >= 0.0 and pymt_disp_txt2.status == NOT_STARTED:
            # keep track of start time/frame for later
            pymt_disp_txt2.tStart = t  # underestimates by a little under one frame
            pymt_disp_txt2.frameNStart = frameN  # exact frame index
            pymt_disp_txt2.setAutoDraw(True)
        
        # *key_resp_pymt_disp* updates
        if t >= 3.0 and key_resp_pymt_disp.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_resp_pymt_disp.tStart = t  # underestimates by a little under one frame
            key_resp_pymt_disp.frameNStart = frameN  # exact frame index
            key_resp_pymt_disp.status = STARTED
            # keyboard checking is just starting
            key_resp_pymt_disp.clock.reset()  # now t=0
            event.clearEvents(eventType='keyboard')
        if key_resp_pymt_disp.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                key_resp_pymt_disp.keys = theseKeys[-1]  # just the last key pressed
                key_resp_pymt_disp.rt = key_resp_pymt_disp.clock.getTime()
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pymt_dispComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()

    #-------Ending Routine "pymt_disp"-------
    for thisComponent in pymt_dispComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_resp_pymt_disp.keys in ['', [], None]:  # No response was made
       key_resp_pymt_disp.keys=None

else: # If S didn't complete the entire binary choice and BDM routines during this session, prompt a manual auction
    # Initialize components for Routine "auc_manual"
    auc_manualClock = core.Clock()
    auc_manual_txt = visual.TextStim(win=win, ori=0, name='auc_manual_txt',
        text=u'Thank you! This completes the experiment. The experimenter will now run the auction and calculate your final payment.\n\nCorrect: ' + str(correct_counter) + '\nIncorrect: ' + str(incorrect_counter),    font=u'Arial',
        pos=[0, 0], height=0.07, wrapWidth=None,
        color=u'white', colorSpace=u'rgb', opacity=1,
        depth=0.0)

    #------Prepare to start Routine "auc_manual"-------
    t = 0
    auc_manualClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    # keep track of which components have finished
    auc_manualComponents = []
    auc_manualComponents.append(auc_manual_txt)
    for thisComponent in auc_manualComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "auc_manual"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = auc_manualClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *auc_manual_txt* updates
        if t >= 0.0 and auc_manual_txt.status == NOT_STARTED:
            # keep track of start time/frame for later
            auc_manual_txt.tStart = t  # underestimates by a little under one frame
            auc_manual_txt.frameNStart = frameN  # exact frame index
            auc_manual_txt.setAutoDraw(True)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in auc_manualComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()

    #-------Ending Routine "auc_manual"-------
    for thisComponent in auc_manualComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

thisExp.nextEntry()

# io.quit() # Close iohub


win.close()
core.quit()
