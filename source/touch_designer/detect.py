
import feature_detection
import osc_sender
from feature_detection import detect_features, render_image
from osc_sender import OSC_Sender
import importlib
import numpy as np
import cv2

sender = OSC_Sender('127.0.0.1', 5000)

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Custom')
	p = page.appendFloat('Valuea', label='Value A')
	p = page.appendFloat('Valueb', label='Value B')
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	return


def to_int(frame):
	return (frame * 255).astype(np.uint8)

def to_float(frame):
	return (frame / 255).astype(np.float32)

def onCook(scriptOp):
	importlib.reload(osc_sender)
	frame = to_int(op('out2').numpyArray(delayed=False))
	features = detect_features(frame)
	sender.send_features(features)
	print(f'num contours: {len(features.contours)}')

	frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

	frame = render_image(frame, features)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	#

	scriptOp.copyNumpyArray(to_float(frame))
