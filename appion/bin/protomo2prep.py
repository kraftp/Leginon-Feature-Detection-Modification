#!/usr/bin/env python

import os
import sys
import math
import glob
from appionlib import appionScript
from appionlib import apDisplay
from appionlib import apDatabase
from appionlib import apProTomo2Prep
from appionlib import apTomo
from appionlib import apProTomo
from appionlib import apParam

#=====================
class ProTomo2Prep(appionScript.AppionScript):
	#=====================
	def setupParserOptions(self):
		self.parser.set_usage( "Usage: %prog --tiltseriesnumber=<#> --session=<session> "
			+"[options]")

		self.parser.add_option("-s", "--session", dest="sessionname",
			help="Session name (e.g. 06mar12a)", metavar="SESSION")

		self.parser.add_option("--tiltseries", dest="tiltseries", type="int",
			help="tilt series number in the session", metavar="int")
			
	#=====================
	def checkConflicts(self):
		pass

		return True

	#=====================
	def onInit(self):
		"""
		Advanced function that runs things before other things are initialized.
		For example, open a log file or connect to the database.
		"""
		return

	#=====================
	def onClose(self):
		"""
		Advanced function that runs things after all other things are finished.
		For example, close a log file.
		"""
		return

	#=====================
	def start(self):
	
		### some of this should go in preloop functions
	
		###do queries
		sessiondata = apDatabase.getSessionDataFromSessionName(self.params['sessionname'])
		self.sessiondata = sessiondata
		tiltseriesdata = apDatabase.getTiltSeriesDataFromTiltNumAndSessionId(self.params['tiltseries'],sessiondata)
		tiltdata=apTomo.getImageList([tiltseriesdata])
		description = self.params['description']
		apDisplay.printMsg("getting imagelist")

		tilts,ordered_imagelist,ordered_mrc_files,refimg = apTomo.orderImageList(tiltdata)
		#tilts are tilt angles, ordered_imagelist are imagedata, ordered_mrc_files are paths to files, refimg is an int

		###set up files
		seriesname='series'+str(self.params['tiltseries'])
		tiltfilename=seriesname+'.tlt'
		param_out=seriesname+'.param'
		maxtilt=max([abs(tilts[0]),abs(tilts[-1])])
		apDisplay.printMsg("highest tilt angle is %f" % maxtilt)
		self.params['cos_alpha']=math.cos(maxtilt*math.pi/180)
		self.params['raw_path']=os.path.join(self.params['rundir'],'raw')


		rawexists=apParam.createDirectory(self.params['raw_path'])

		apDisplay.printMsg("copying raw images")
		newfilenames=apProTomo.getImageFiles(ordered_imagelist,self.params['raw_path'], link=False)
	
		###create tilt file

		#get image size from the first image
		imagesizex=tiltdata[0]['image'].shape[0]
		imagesizey=tiltdata[0]['image'].shape[1]

		#shift half tilt series relative to eachother
		#SS I'm arbitrarily making the bin parameter here 1 because it's not necessary to sample at this point
		shifts = apTomo.getGlobalShift(ordered_imagelist, 1, refimg)
		
		#OPTION: refinement might be more robust by doing one round of IMOD aligment to prealign images before doing protomo refine
		origins=apProTomo2Prep.convertShiftsToOrigin(shifts, imagesizex, imagesizey)

		#determine azimuth
		azimuth=apTomo.getAverageAzimuthFromSeries(ordered_imagelist)
		apProTomo2Prep.writeTileFile2(tiltfilename, seriesname, newfilenames, origins, tilts, azimuth, refimg)

#=====================
if __name__ == '__main__':
	protomo2prep = ProTomo2Prep()
	protomo2prep.start()
	protomo2prep.close()

