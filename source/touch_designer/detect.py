
import feature_detection
from feature_detection import detect_features, render_image
from osc_sender import OSC_Sender
import importlib
import numpy as np

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


def onCook(scriptOp):
	importlib.reload(feature_detection)
	np_arr = op('displace2').numpyArray(delayed=True)
	print(f'form = {np_arr.astype(np.uint8).shape}')

	features = detect_features(np_arr.astype(np.uint8))
	print(f'len = {len(features.contours)}')
	sender.send_features(features)

	scriptOp.copyNumpyArray(np_arr)
