

# imports:

# pythonocc, used to interface with 3d models
import OCC
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_QuaternionSLerp, gp_Quaternion, gp_Vec, gp_Pnt, gp_Trsf, gp_Ax1, gp_Dir, gp_GTrsf, gp_Pnt2d, gp_XOY, gp_Circ, gp_Ax2, gp_XYZ
from OCC.Extend.ShapeFactory import translate_shp, rotate_shp_3_axis
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism 
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform, BRepBuilderAPI_MakeShell, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeEdge2d, BRepBuilderAPI_MakeSolid
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Extend.DataExchange import read_step_file, read_iges_file, read_stl_file, write_step_file
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_CompCurve, BRepAdaptor_Surface
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepExtrema import BRepExtrema_ShapeProximity, BRepExtrema_DistShapeShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.BOPAlgo import BOPAlgo_Builder
from OCC.Core.Geom import Geom_TrimmedCurve, Geom_OffsetCurve, Geom_Line, Geom_OffsetSurface
from OCC.Core.GeomAdaptor import geomadaptor_MakeCurve
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset, BRepOffsetAPI_MakeOffsetShape, BRepOffsetAPI_MakePipeShell, BRepOffsetAPI_MakePipe
from OCC.Core.GeomAbs import GeomAbs_Arc, GeomAbs_Tangent, GeomAbs_Intersection
from OCC.Core.BRepOffset import BRepOffset_Skin
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface, ShapeAnalysis_FreeBounds 
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepOffset import BRepOffset_MakeSimpleOffset, BRepOffset_MakeOffset
from OCC.Core.Geom2d import Geom2d_OffsetCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TopExp import TopExp_Explorer, topexp_FirstVertex, topexp_LastVertex
from OCC.Core.ShapeAnalysis import ShapeAnalysis_WireOrder, ShapeAnalysis_Curve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.ShapeFix import ShapeFix_ComposeShell
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.TopAbs import (
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_WIRE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_ShapeEnum,
)
from OCC.Core.GeomAbs import (
	GeomAbs_Line,
	GeomAbs_Circle,
	GeomAbs_Ellipse,
	GeomAbs_Hyperbola,
	GeomAbs_Parabola,
	GeomAbs_BezierCurve,
	GeomAbs_BSplineCurve,
	GeomAbs_OffsetCurve,
	GeomAbs_OtherCurve,
)

# additional external libraries
from distinctipy import distinctipy # distinct color generator for visualizations
import numpy as np # used for parallel vector calculations
import networkx as nx # used to represent the 3d object as a 2d graph, and extract paths from said graph
import psutil # used to record ram usage
from memory_profiler import memory_usage # used to record ram usage
from numba import jit, cuda # gpu acceleration
from numba.typed import Dict as numbaDict # gpu acceleration
from numba import types # gpu acceleration
import pyvista as pv # surface reconstruction
import open3d as o3d # point cloud outlier removal
import pymeshfix as mf # mesh fixing, not sure if needed/will work
import pyacvd # remeshing, not sure if needed/will work
#import pymesh # remeshing, not sure if needed/will work

# custom libraries
from windowFunctions import setupControls # places custom window control functions in a seperate file
from poolQueue import PoolQueue # easier queue based multiprocessing

# standard python libraries
import time # used for timing functions
import math # used primarily for rounding coordinates
import os # used for io, and process identification
import cProfile # base CPython profiler
import queue # needed primarily for the queue.Empty exception
import pickle # used to serialize objects to send data between threads

#

white = Quantity_Color(1, 1, 1, Quantity_TOC_RGB)
black = Quantity_Color(0, 0, 0, Quantity_TOC_RGB)

red = Quantity_Color(1, 0, 0, Quantity_TOC_RGB)
green = Quantity_Color(0, 1, 0, Quantity_TOC_RGB)
blue = Quantity_Color(0, 0, 1, Quantity_TOC_RGB)

yellow = Quantity_Color(1, 1, 0, Quantity_TOC_RGB)
magenta = Quantity_Color(1, 0, 1, Quantity_TOC_RGB)
cyan = Quantity_Color(0, 1, 1, Quantity_TOC_RGB)

"""
g = nx.DiGraph()

g.add_node(1)
g.add_node(2)
g.add_node(3)

g.add_edge(1, 2)

idk = list(g.out_edges(1, data=True))
print(idk)
print(idk[0])

print(len(g.out_edges(1, data=True)))
print(len(g.out_edges(3, data=True)))

exit(1)

"""

#

def getCurrentRamUsage(num = None, suffix = "B"):

	# this function only shows memory usage for the main thread 
	# not for multiprocessing

	if num is None:
		process = psutil.Process(os.getpid())
		num = process.memory_info().rss
	
	# https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
	for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
		if abs(num) < 1024.0:
			return f"{num:3.1f}{unit}{suffix}"
		num /= 1024.0
	return f"{num:.1f}Yi{suffix}"

def getFunctions(obj):
	# debugging method, just to show what functions can be called on any given object
	res = [method_name for method_name in dir(obj) if callable(getattr(obj, method_name))]
	res = [x for x in res if not x.startswith("_")]
	return res

#

class TopoDS_Wrapper:

	# the following code is stupid.
	# i am reaching conspiracy level bs with pythonocc's hashes and 
	# am now rewriting the dict class
	# nope, instead just going to make a wrapper for TopoDS objects
	
	# important note, figure out how much this impacts performance
	# it may be possible to overload the hash functions of the non wrapped 
	# objects 
	
	# https://wiki.python.org/moin/UsingSlots
	__slots__ = ("obj")
	
	def __new__(cls, obj):
	
		# this error checking should not be here in prod 
		if not issubclass(type(obj), OCC.Core.TopoDS.TopoDS_Shape):
			print("TopoDS_Wrapper may only wrap actual TopoDS shapes! not " + str(type(obj)))
			exit(1)
			
		if type(obj) == OCC.Core.TopoDS.TopoDS_Edge:
			return object.__new__(_TopoDS_EdgeWrapper)
		
		if type(obj) == OCC.Core.TopoDS.TopoDS_Wire:
			# IS THIS ACCEPTABLE
			return object.__new__(_TopoDS_EdgeWrapper)
			
		return object.__new__(TopoDS_Wrapper)
	
	def __init__(self, obj):
		self.obj = obj
	
	def __hash__(self):
		return self.obj.HashCode((1 << 31) - 1)
		
	def getHash(self):
		# just for debugging purposes
		return self.__hash__()
		
	def __eq__(self, other):
		return self.__hash__() == other.__hash__()

	def get(self):
		return self.obj

class _TopoDS_EdgeWrapper(TopoDS_Wrapper):

	# https://wiki.python.org/moin/UsingSlots
	# bad code
	__slots__ = ("obj", "faceHashes", "startParameter", "endParameter", "startPoint", "endPoint", "startVector", "endVector", "closed", "length", "curveType")

	def __new__(self, obj=None):
		return object.__new__(_TopoDS_EdgeWrapper)
	
	def _doInit(self, obj):
		
		self.obj = obj
	
		self.faceHashes = set()
		
		#curve = BRepAdaptor_Curve(self.obj)
		
		if type(obj) == OCC.Core.TopoDS.TopoDS_Edge:
			curve = BRepAdaptor_Curve(self.obj)
		
		if type(obj) == OCC.Core.TopoDS.TopoDS_Wire:
			curve = BRepAdaptor_CompCurve(self.obj)
		
		# https://dev.opencascade.org/doc/refman/html/_geom_abs___curve_type_8hxx.html#af25c179d5cabd33fddebe5a0dc96971c
		self.curveType = curve.GetType()
		
		# this rounding is extremely dumb and risky.
		# FIX: FIND BETTER OPTION TO MAKE SURE EDGE START AND ENDPOINTS ALWAYS CONNECT
		roundPoint = lambda cord : tuple([ round(n, 2) for n in cord])
		
		self.startParameter, self.endParameter = curve.FirstParameter(), curve.LastParameter()
		self.startPoint, self.endPoint = roundPoint(curve.Value(self.startParameter).Coord()), roundPoint(curve.Value(self.endParameter).Coord())
		self.startVector, self.endVector = curve.DN(self.startParameter, 1).Coord(), curve.DN(self.endParameter, 1).Coord()
		self.closed = curve.IsClosed()
		
		self.length = GCPnts_AbscissaPoint().Length(curve)
	
	def __init__(self, obj):
		self._doInit(obj)
	
	def __repr__(self):
		return "Edge:" + str(self.startPoint) + "-->" + str(self.endPoint)
	
	def reCalculate(self):
		prevFaceHashes = self.faceHashes
		self._doInit(self.obj)
		self.faceHashes = prevFaceHashes
		
	def getFaceHashes(self):
		return self.faceHashes
		
	def hasFaceHash(self, hash):
		return hash in self.faceHashes

	def addFaceHashes(self, hashes):
		self.faceHashes.update(hashes)
	
	def getStartPoint(self):
		return self.startPoint
	
	def getEndPoint(self):
		return self.endPoint
	
	def isClosed(self):
		return self.closed
	
	def hasPoint(self, p):
		# returns boolean of if the point p is either the start or end point
		return (p == self.startPoint) or (p == self.endPoint)
	
	def getPointVector(self, p):
		# get the vector at the point p
		# MUST BE EITHER THE START OR END POINT

		if not self.hasPoint(p):
			print("point not in edge")
			exit(1)
		
		if p == self.startPoint:
			return self.startVector
		
		if p == self.endPoint:
			return self.endVector
		
		return None
		
	def isParallel(self, otherEdgeWrapper, p, returnAngle = False, tolerate = False):
		
		# THIS HAS BEEN CHANGED DURING STL STUFF AND MAY NEED TO BE 
		# ADJUSTED FOR STEP 
		
		# ok, looking back on this code months after. 
		# what the is this 
		# it doesnt even really get a angle???
		# what the does this do????
		
		if type(otherEdgeWrapper) != _TopoDS_EdgeWrapper:
			print("isParallel was a non edge wrapper")
			exit(1)
		
		if not tolerate:
			if not (self.hasPoint(p) and otherEdgeWrapper.hasPoint(p)):
				print("isParallel other did not share the point.")
				print("self points:")
				print("start: " + str(self.getStartPoint()) + ", end: " + str(self.getEndPoint()))
				print("other points:")
				print("start: " + str(otherEdgeWrapper.getStartPoint()) + ", end: " + str(otherEdgeWrapper.getEndPoint()))
				print("the point in question: " + str(p))
				exit(1)
			selfVector = self.getPointVector(p)
			otherVector = otherEdgeWrapper.getPointVector(p)
		else:
		
			pointIndices = [
			[0, 0],
			[0, 0],
			]
			
			testPoint = np.array(p)
			
			for i, edgeWrapper in enumerate([self, otherEdgeWrapper]):
				points = [np.array(edgeWrapper.startPoint), np.array(edgeWrapper.endPoint)]
				
				#points[0] = np.sum(np.abs(points[0] - testPoint))
				#points[1] = np.sum(np.abs(points[0] - testPoint))
				
				points[0] = np.sum(np.power(points[0] - testPoint, 2))
				points[1] = np.sum(np.power(points[0] - testPoint, 2))
				
				if points[0] < points[1]:
					pointIndices[i][0] = 1
				else:
					pointIndices[i][1] = 1
		
			selfVector = self.startVector if pointIndices[0][0] else self.endVector
			otherVector = otherEdgeWrapper.startVector if pointIndices[1][0] else otherEdgeWrapper.endVector
			
		
		
		# could cast to the vec class and go from there, but instead 
		# just going to do that math on numpy
		
		if returnAngle:
			#return np.linalg.norm(np.cross(selfVector, otherVector))
			
			# THESE VECS SHOULD REALLY BE NORMALIZED LIKE IN THE ACTUAL THING
			# but i think that them not being normalized is what makes 
			# the initial parallel code work???
			v1_u = selfVector / np.linalg.norm(selfVector)
			v2_u = otherVector / np.linalg.norm(otherVector)
			angle = angle = 180 / np.pi * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 
			return angle
		
		# FOR STEP USE 0.1, FOR STL USE 5, OR SOMETHING ELSE
		# i have 0 CLUE what this is but im keeping this in for legacy sake
		#return np.linalg.norm(np.cross(selfVector, otherVector)) < 0.1	
		#return angle < 10
		return np.linalg.norm(np.cross(selfVector, otherVector)) < 5	

	def _getSharedPoint(self, otherEdgeWrapper):
		
		# this code is getting horrible.
		# this shit NEEEDS ERROR CHECKING IN CASE I EVER USE IT ELSEWHERE GOD
		# underscored until then
		
		res = [None] * 4
		
		for i, p1 in enumerate([self.startPoint, self.endPoint]):
			for j, p2 in enumerate([otherEdgeWrapper.startPoint, otherEdgeWrapper.endPoint]):
				res[j * 2 + i] = np.sum(np.power(np.array(p1) - np.array(p2), 2))
		
		resIndex = res.index(min(res)) % 2
		
		return [self.startPoint, self.endPoint][resIndex]
		
	def getAngle(self, otherEdgeWrapper):
		# so much of the shit in here could be put into ispar.
		
		if type(otherEdgeWrapper) != _TopoDS_EdgeWrapper:
			print("isParallel was a non edge wrapper")
			exit(1)
		
		pointIndices = [
			[0, 0],
			[0, 0],
		]
		
		p = self._getSharedPoint(otherEdgeWrapper)
		
		testPoint = np.array(p)
		
		for i, edgeWrapper in enumerate([self, otherEdgeWrapper]):
			points = [np.array(edgeWrapper.startPoint), np.array(edgeWrapper.endPoint)]
			
			points[0] = np.sum(np.power(points[0] - testPoint, 2))
			points[1] = np.sum(np.power(points[1] - testPoint, 2))
			
			if points[0] < points[1]:
				pointIndices[i][0] = 1
			else:
				pointIndices[i][1] = 1
	
		selfVector = self.startVector if pointIndices[0][0] else self.endVector
		otherVector = otherEdgeWrapper.startVector if pointIndices[1][0] else otherEdgeWrapper.endVector

		v1_u = selfVector / np.linalg.norm(selfVector)
		v2_u = otherVector / np.linalg.norm(otherVector)
		angle = angle = 180 / np.pi * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 
		
		return angle

#

def trimEdge(edge, start, end):
	
	# https://github.com/tpaviot/pythonocc-core/issues/1057
	# what in the hell is this
	
	# todo, replace all instances of this code with this function
	curve = BRepAdaptor_Curve(edge.get())
	
	
	#if curve.GetType() == 7:
	#	print("WEIRD CURVE, DIPPING")	
	#	return edge
					
	curveHandle = curve.Curve()
	tempCurve = curveHandle.Trim(start, end, 0.1)
	tempEdge = geomadaptor_MakeCurve(tempCurve)
	tempEdge = BRepBuilderAPI_MakeEdge(tempEdge).Edge()
					
	# i absolutely love dealing with C++ style memory in python 
	# there is 0 documentation on this. pray
	# i literally guessed this, and had it work
	curveHandle.DecrementRefCounter()   

	#res = TopoDS_Wrapper(tempEdge)
	# skip a check, why not
	res = _TopoDS_EdgeWrapper(tempEdge)
	
	#print(curve.GetRefCount(), curveHandle.GetRefCount(), tempCurve.GetRefCount())
	
	res.addFaceHashes(edge.getFaceHashes())

	return res

def rotateVector(v, axis, angle):

	# this could use optimization 
	# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

	s = np.cos(angle / 2 * np.pi / 180)

	u = np.array(axis) * np.sin(angle / 2 * np.pi / 180)

	res = 2 * np.dot(u, v) * u + (s*s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)

	return res
	
def getNormalVector(edge, hashToFace, returnBoth = False, lineProgress = None):

	# THIS BREP STUFF COULD (AND MAYBE SHOULD) BE DONE IN _TopoDS_EdgeWrapper
	# but with reversing and offseting, (mostly reversing) problems may occur
	curve = BRepAdaptor_Curve(edge.get())
	
	tool = BRep_Tool()

	faces = [ hashToFace[hash] for hash in edge.getFaceHashes() ]

	normals = []
	orientations = []
	for face in faces:
		faceHandle = tool.Surface(face.get())
		
		uvPoint = gp_Pnt2d(0, 0)
		
		shapeAnalyzer = ShapeAnalysis_Surface(faceHandle)
		# the 2 and 0.1 params here are important, but i have 0 clue what they do
		normalFinder = GeomLProp_SLProps(faceHandle, 2, 0.1)
		
		if lineProgress is None:
			tempTestPoint = gp_Pnt(*edge.startPoint)
		else:
			tempTestPoint = curve.Value( (edge.endParameter - edge.startParameter) * lineProgress )

		res = shapeAnalyzer.NextValueOfUV(uvPoint, tempTestPoint, 1)
		
		normalFinder.SetParameters(*res.Coord())
	
		n = normalFinder.Normal().Coord()

		orientation = face.get().Orientation()
		orientations.append(orientation)
		
		if orientation == 0:
			pass
		elif orientation == 1:
			n = np.negative(n)
		else:
			print("unknown face orientation of {:d} encountered. figure it out".format(orientation))
			exit(1)
		
		normals.append(n)
	
	normalVector = np.divide(np.add(normals[0], normals[1]), 2)
	normalVector = normalVector / np.linalg.norm(normalVector)
		
	if returnBoth:
		return normals
		
	return normalVector

def getOffsetVector(edge, hashToFace):

	# this could be put into the edgewrap class, but there 
	# are some issues with pickling faces for some reason.

	# this used to return normal, will now be getoffset vector.
	# basically, we get the normal, and then we need to rotate it such that 
	# it faces where the tool should be when at a 45 deg to the surface 
	# as in the tool will connect with the main edge, and the edge from this offset 
	# to create the break edge
	# in order to do this, we must rotate the vector around its edge.
	# matrix stuff, wonderful


	normalVector = getNormalVector(edge, hashToFace)
	
	# rotate the normal vector

	# for some reason, the direct call to curve.DN done in _TopoDS_EdgeWrapper 
	# does not return normalized vectors???
	axis = edge.startVector / np.linalg.norm(edge.startVector)
	
	# bad code
	# there is totally a better way of doing this
	offsetVector = rotateVector(normalVector, axis, 1)
	
	zDif = offsetVector[2] - normalVector[2]
	
	angle = 0
	
	if zDif >= 0:
		#display.DisplayShape(edge.get(), color=yellow)
		offsetVector = rotateVector(normalVector, axis, 270)
		angle = 270
	else:
		#display.DisplayShape(edge.get(), color=cyan)
		offsetVector = rotateVector(normalVector, axis, 90)
		angle = 90
	
	
	"""
	# this might be better, but is still VERY BAD
	# i do not know how it works/what it does 
	offsetVector = rotateVector(normalVector, axis, 1)
	
	difVector = offsetVector - normalVector
	
	difVector = np.abs(difVector)
	
	if min(difVector) == difVector[0]:
		display.DisplayShape(edge.get(), color=cyan)
		offsetVector = rotateVector(normalVector, axis, 90)
		angle = 90
	elif min(difVector) == difVector[1]:
		display.DisplayShape(edge.get(), color=yellow)
		offsetVector = rotateVector(normalVector, axis, 90)
		angle = 90
	else: # max(difVector) == difVector[2]:
		display.DisplayShape(edge.get(), color=magenta)
		#offsetVector = rotateVector(normalVector, axis, 270)
		#angle = 270
		offsetVector = rotateVector(normalVector, axis, 90)
		angle = 90
	"""
	
	#return normalVector, 0
	return offsetVector, angle

def getToolAngles(v):
	
	#v = N(v)
	v = v / np.linalg.norm(v)
	
	p1 = v
	p2 = np.array([0, 0, 1])
	
	xy = 180/np.pi * ( np.arctan2(*p1[[1, 0]]) - np.arctan2(*p2[[1, 0]]) )
	xz = 180/np.pi * ( np.arctan2(*p1[[2, 0]]) - np.arctan2(*p2[[2, 0]]) )
	yz = 180/np.pi * ( np.arctan2(*p1[[2, 1]]) - np.arctan2(*p2[[2, 1]]) )

	z = xy
	y = xz
	x = yz

	#res = np.array([xy, xz, yz])
	#res = np.array([x, y, z])
	res = np.array([x, y])
	res += 180
	
	#print(x, y, z)

	res[0] = -res[0]
	res[1] = -res[1]
	
	res = res / np.linalg.norm(res)
	res *= 45

	return res

def getToolTarget(edge, offsetEdge, parameter):

	edgeCurve = BRepAdaptor_Curve(edge.get())
	offsetCurve = BRepAdaptor_Curve(offsetEdge.get())

	edgePoint = np.array(edgeCurve.Value(parameter).Coord())
	offsetPoint = np.array(offsetCurve.Value(parameter).Coord())
							
	offsetVector = offsetPoint - edgePoint
	offsetVector = offsetVector / np.linalg.norm(offsetVector)
					
	toolAngles = getToolAngles(offsetVector)
	
	return offsetPoint, toolAngles

def reverseEdge(edge):
	
	curve = BRepAdaptor_Curve(edge.get())
	
	curveHandle = curve.Curve()
	geomCurve = curveHandle.Curve()
	reversedCurve = geomCurve.Reversed()
	
	tempEdge = BRepBuilderAPI_MakeEdge(reversedCurve).Edge()
	
	res = _TopoDS_EdgeWrapper(tempEdge)
	res.addFaceHashes(edge.getFaceHashes())

	# satan
	curveHandle.DecrementRefCounter()
	
	# when making any modifications at the curve level, things need to be retrimmed.
	# however, due to infinite curves, memory leaks, and other bs 
	# this is the best way i know of doing this 
		
	pointProjector = GeomAPI_ProjectPointOnCurve()
	paramDelta = max(abs(edge.startParameter), abs(edge.endParameter)) + (edge.endParameter - edge.startParameter) / 2
	pointProjector.Init(reversedCurve, -paramDelta, paramDelta)
	
	pointProjector.Perform(gp_Pnt(*edge.startPoint))
	newEnd = pointProjector.LowerDistanceParameter()
	
	pointProjector.Perform(gp_Pnt(*edge.endPoint))
	newStart = pointProjector.LowerDistanceParameter()
	
	res2 = trimEdge(res, newStart, newEnd)

	curveHandle.DecrementRefCounter()
	
	return res2

def orderPath(path):

	if len(path) == 1:
		return path

	wireOrderer = ShapeAnalysis_WireOrder()
	
	edgeMap = {} # maps the edge indices to actual edges
	
	i = 1
	for edgePair in path:
		
		edge, offsetEdge = edgePair
		
		wireOrderer.Add(gp_XYZ(*edge.startPoint), gp_XYZ(*edge.endPoint))
		
		edgeMap[i] = edgePair
		i += 1
		
	if wireOrderer.NbChains() != 0:
		print("when attempting to order wires, multiple chains were created. this should never happen")
		exit(1)
		
	wireOrderer.Perform()

	res = []
	
	for i in range(1, len(path)+1):
		
		order = wireOrderer.Ordered(i)
		
		edgeMapData = edgeMap[abs(order)]
		edge, offsetEdge = edgeMapData
		
		if order < 0:
			edge = reverseEdge(edge)
			offsetEdge = reverseEdge(offsetEdge)
		
			edgeMap[abs(order)] = [edge, offsetEdge]
		
		res.append(edgeMap[abs(order)])
		
	#res = [ v[1] for v in sorted(edgeMap.items(), key=lambda o : o[0], reverse = True) ]
		
	return res

def combineSTLPath(path):

	# this is most likely NOT the best solution.
	# but it will work 
	
	# it will have some issues with the starts and ends of curves
	
	# THIS MAY BE A BAD IDEA
	for i in range(0, len(path)):
		path[i][0].reCalculate()
		path[i][1].reCalculate()
	
	# THIS MAY BE AN ISSUE
	if len(path) <= 3:
		# took me way to long to remember this edge case and like actually put the false in here
		return [[False, path]]
	
	roundPoint = lambda p : tuple(np.round(p, 1))
	
	graph = nx.DiGraph()
	
	for edgePair in path:

		edge, offsetEdge = edgePair
		
		# this rounding may be a bad idea, but there are 
		# some cases where the points are off by 0.01 (for unknown reasons)
		
		startPoint = roundPoint(edge.startPoint)
		endPoint = roundPoint(edge.endPoint)
		
		for p in [startPoint, endPoint]:
			if p not in graph.nodes:
				graph.add_node(p)
				graph.nodes[p]["edgeData"] = []
		
		graph.add_edge(startPoint, endPoint)
		graph.edges[startPoint, endPoint]["edge"] = edgePair

	if len(list(nx.weakly_connected_components(graph))) != 1:
		print("somehow, a digraph of a path was not all connected. this is very bad")
		exit(1)

	for i in range(1, len(path) - 1):
		
		prevEdgePair = path[i - 1] 
		edgePair = path[i]
		nextEdgePair = path[i + 1]
		
		prevEdge = prevEdgePair[0]
		edge = edgePair[0]
		nextEdge = nextEdgePair[0]
		
		prevAngle = prevEdge.getAngle(edge)
		nextAngle = edge.getAngle(nextEdge)
		
		angleDif = nextAngle - prevAngle
		
		if abs(angleDif) > 5:
			if angleDif < 0:
				edgeNodes = (roundPoint(edge.startPoint), roundPoint(edge.endPoint))
				graph.remove_edge(*edgeNodes)
				graph.nodes[edgeNodes[1]]["edgeData"].append(edgePair)
				#display.DisplayShape(edge.get(), color=magenta)
				pass
			else:
				edgeNodes = (roundPoint(edge.startPoint), roundPoint(edge.endPoint))
				graph.remove_edge(*edgeNodes)
				graph.nodes[edgeNodes[0]]["edgeData"].append(edgePair)
				#display.DisplayShape(edge.get(), color=yellow)
		else:
			#display.DisplayShape(edge.get(), color=red)
			pass
			
	groups = list(nx.weakly_connected_components(graph))
	
	res = []
	
	colors = loadColors()
	colorIndex = 0
	
	for subPath in groups:
		
		colorIndex += 1
		
		tempRes = []	
	
		for node in subPath:
			
			if len(graph.nodes[node]["edgeData"]) != 0:
				#tempRes.append(graph.nodes[node]["edgeData"][0])
				for pair in graph.nodes[node]["edgeData"]:
					tempRes.append(pair)
				
			nodeEdges = list(graph.out_edges(node, data = True))
			if len(nodeEdges) == 1:
				tempRes.append(nodeEdges[0][2]["edge"])
			
		if len(tempRes) == 0:
			continue
			
		tempResPath = orderPath(tempRes)
		
		subPaths = []
		
		if len(tempResPath) == 1:
			subPaths.append([False, tempResPath])
		else:
			
			pathState = None
			currentSubPath = []
			
			for i in range(0, len(tempResPath) - 1):
				
				edge = tempResPath[i][0]
				nextEdge = tempResPath[i + 1][0]
				
				angle = edge.getAngle(nextEdge)
				
				# true if circle, false if line
				lineType = angle > 5
				
				if pathState is None:
					pathState = lineType
					currentSubPath = [tempResPath[i]]
					continue
				
				if pathState == lineType:
					# continue current seg type
					currentSubPath.append(tempResPath[i])
				else: # start new seg type
					subPaths.append([pathState, currentSubPath])
					pathState = lineType
					currentSubPath = [tempResPath[i]]
			
			currentSubPath.append(tempResPath[-1])
			subPaths.append([pathState, currentSubPath])
		
		res.extend(subPaths)

	return res
	
def generateColors():
	
	# generate colors, pickle them, and then save to a file 
	# i would generate them every time, but it takes a while
	
	print("generating colors")
	print("this is a one time function, and will not be ran every time.")
	
	numColors = 64
	
	colors = distinctipy.get_colors(numColors)
	
	with open("colors.pickle", "wb") as f:
		pickle.dump(colors, f)
	
	print("colors saved to file")

def loadColors():
	
	if hasattr(loadColors, "res"):
		return loadColors.res
	
	if not os.path.exists("colors.pickle"):
		generateColors()
		
	with open("colors.pickle", "rb") as f:
		colors = pickle.load(f)
	
	loadColors.res = [ rgb_color(*c) for c in colors ]	
	
	return loadColors.res

#

def pointCloudToStep():
	
	# convert point cloud to mesh file
	
	"""
	doNormals = lambda obj : obj.compute_normals(auto_orient_normals=True, flip_normals=False, progress_bar = True)
	
	print("removing outliers from point cloud")
	
	#surface = pv.read("Top Plate 3DInfotech.stl")
	#surface = pv.read("scan4_convex.stl")
	#surface = pv.read("Scan 4.stl")
	surface = pv.read("untitled.stl")
	
	
	truncate = lambda n, prec : np.true_divide(np.floor(n * 10 ** prec), 10 ** prec)
	#roundFunc = lambda p : (np.round(p[0], 3), np.round(p[1], 3), truncate(p[2], 1))
	
	# this one was being used, andwas good
	#roundFunc = lambda p : (np.round(p[0], 3), np.round(p[1], 3), np.round(p[2], 1))
	
	roundFunc = lambda p : (np.round(p[0], 3), np.round(p[1], 3), np.round(p[2], 0))
	
	surface.points = np.apply_along_axis(roundFunc, 1, surface.points)
	
	# am not sure if this should be here !!!!!
	surface.clean()
	
	surface.save("scan0Rounded.stl")
	
	
	mesh = mf.PyTMesh()
	mesh.load_file("scan0Rounded.stl")
	#mesh.fill_small_boundaries()
	#mesh.fill_small_boundaries(refine = True)
	#mesh.fill_small_boundaries(nbe=100, refine=True)
	#mesh.clean(max_iters=10, inner_loops=3)
	
	print('There are {:d} boundaries'.format(mesh.boundaries()))
	
	mesh.remove_smallest_components()
	mesh.join_closest_components()
	
	mesh.save_file("scan0TestFile.stl")
	
	# this is a weird alternative, idk if its good or not
	#surface = pv.read("scan0TestFile.stl")
	#clus = pyacvd.Clustering(surface)
	#print("subdividing")
	#clus.subdivide(2)
	#print("clustering")
	#clus.cluster(30000)
	#remesh = clus.create_mesh()
	#remesh.save("scan0TestFileOutput.stl")
	
	time.sleep(0.1) # pure conspiracy
	surface = pv.read("scan0TestFile.stl")
	#surface = pv.read("scan0TestFileOutput.stl")
	
	
	# only for the stl from company, not point cloud
	surface = surface.fill_holes(100) # I HAD THIS OFF FOR SO LONG AND IT WAS CAUSING MY PROBLEMS GOD I AM DUMB
	#surface = doNormals(surface)
	
	
	#doDecimatePro = lambda obj, percent : obj.decimate_pro(percent, feature_angle = 45, progress_bar = True)
	doDecimatePro = lambda obj, percent : obj.decimate_pro(percent, 
	feature_angle = 89.99,
	#split_angle = 75, 
	#splitting = False, 
	#pre_split_mesh = True,
	#preserve_topology = True,
	#max_degree = 10,
	progress_bar = True)
	
	doDecimateSimple = lambda obj, percent : obj.decimate(percent,
	volume_preservation = True, 
	#volume_preservation = False, 
	#attribute_error = True,
	progress_bar = True)


	print("mesh faces: ", surface.n_faces)
	percent = 0.50
	#percent = 0.75
	#percent = 0.95
	print("doing a reduction of ", percent)	
	surface = doDecimatePro(surface, percent)
	
	# THIS SHOULDNT BE HERE FOR FULL SCAN RUNS
	#surface = doNormals(surface)
	
	print("mesh faces: ", surface.n_faces)
	
	if surface.n_faces > 1000:
		print("mesh faces: ", surface.n_faces)
		#percent = (surface.n_faces - 3000) / surface.n_faces
		#percent = (surface.n_faces - 2000) / surface.n_faces
		#percent = (surface.n_faces - 500) / surface.n_faces
		percent = (surface.n_faces - 1000) / surface.n_faces
		#percent = (surface.n_faces - 600) / surface.n_faces
		#percent = (surface.n_faces - 2000) / surface.n_faces
		print("doing a reduction of ", percent)	
		#surface = surface.decimate_boundary(percent)	
		surface = doDecimateSimple(surface, percent)
		#surface = doDecimatePro(surface, percent)
		#surface = doNormals(surface)
		print("mesh faces: ", surface.n_faces)
	
	
	surface.save("scan0.stl")
	
	#exit(0)
	"""
	
	time.sleep(0.1) # pure conspiracy
	compound = read_stl_file("scan0.stl")
	
	sewingBuilder = BRepBuilderAPI_Sewing()
	sewingBuilder.Load(compound)
	sewingBuilder.Perform()
	compound = sewingBuilder.SewedShape()

	t = TopologyExplorer(compound)
	print("edge", t.number_of_edges())
	print("face", t.number_of_faces())
	print("solid", t.number_of_solids())
	print("shells", t.number_of_shells())
	
	maxFaceCount = 0
	
	shells = list(t.shells())
	
	shell = None
	
	for s in shells:
		tempT = TopologyExplorer(s)
		
		if tempT.number_of_faces() > maxFaceCount:
			shell = s
			maxFaceCount = tempT.number_of_faces()
	
	compound = BRepBuilderAPI_MakeSolid(shell).Shape()
	
	t = TopologyExplorer(compound)

	if t.number_of_solids() != 1:
		print("point cloud to step couldnt properly generate a solid!")
		exit(1)
		
	#display.DisplayShape(compound)
	
	#write_step_file(list(t.solids())[0], "scan0.stp")
	#time.sleep(0.1)
	
	#stepFile = read_step_file("scan0.stp")
	#display.DisplayShape(stepFile)
	
	res = t
	
	print("converted point cloud to step")
	
	
	
	return res

def loadFile(filename):
	
	compound = read_step_file(filename)

	t = TopologyExplorer(compound)
	
	if t.number_of_solids() != 1:
		print("This program only works with models containing 1 solid, not " + str(t.number_of_solids()))
		exit(1)
	
	# this step is important
	# removes random edges that are not part of the solid
	t = TopologyExplorer(list(t.solids())[0])
	
	return t

def getFilteredEdges(t):

	# in step models, some models will have extranious edges 
	# in the areas between faces, or along cylinders.
	# this function removes those edges.

	if type(t) != OCC.Extend.TopologyUtils.TopologyExplorer:
		print("getFilteredEdges REQUIRES an object of type OCC.Extend.TopologyUtils.TopologyExplorer")
		print("not a " + str(type(t)))
		exit(1)
	
	# map. key is the hash of the EDGE. 
	# value is array of format (continuity, face1, face2) \
	# first element of the array is a INT indicating its continuity.
	# https://dev.opencascade.org/doc/refman/html/_geom_abs___shape_8hxx.html#a943632453b69386bece6c091156b1ed5
	
	faceEdgesContinuity = {}
	
	# configure data 
	for face in t.faces():
		# another topology explorer here may be faster, do some testing
		
		wrappedFace = TopoDS_Wrapper(face)
		
		for edge in t.edges_from_face(face):
		
			wrappedEdge = TopoDS_Wrapper(edge)
		
			if wrappedEdge not in faceEdgesContinuity:
				faceEdgesContinuity[wrappedEdge] = [-1]
				
			faceEdgesContinuity[wrappedEdge].append(wrappedFace)
	
	# set continuity 
	bt = BRep_Tool()
	for edge in faceEdgesContinuity.keys():
	
		if len(faceEdgesContinuity[edge]) == 2:
			# only one face had this edge
			continue
	
		if len(faceEdgesContinuity[edge]) != 3:
			print("somehow, an edge had more than 2 faces bordering it? it had " + str(len(faceEdgesContinuity[edge])))			
			exit(1)
			
		face1, face2 = faceEdgesContinuity[edge][1:]	
		
		tempRes = bt.HasContinuity(edge.get(), face1.get(), face2.get())
		if tempRes:
			faceEdgesContinuity[edge][0] = bt.Continuity(edge.get(), face1.get(), face2.get())
	
	
	# for the alternative path gen 
	# if faces have continuity, link them together
	# record what faces have continuity
	# key is the face, val is the array of faces that it has continuity with
	graph = nx.Graph()
	
	res = []
	for k, v in faceEdgesContinuity.items():
	
		if v[0] == 0 and len(v) == 3:
			#k.addFaces(v[1:])
			#k.addFaces([v[1], v[2]])
			k.addFaceHashes([v[1].__hash__(), v[2].__hash__()])
			res.append(k)
			graph.add_node(v[1])
			graph.add_node(v[2])
			#graph.add_node(v[1].__hash__())
			#graph.add_node(v[2].__hash__())
		elif v[0] == 1 and len(v) == 3: # len(v[1:]) == 2

			# add node for each and an edge
			for face in v[1:]:
				if face not in graph.nodes:
					graph.add_node(face)
					#graph.add_node(face.__hash__())
			
			graph.add_edge(v[1], v[2])			
			#graph.add_edge(v[1].__hash__(), v[2].__hash__())
			
			
	# get connected face groups
	
	faceGroups = list(nx.connected_components(graph))
	
	linkedFaces = set()
	#linkedFaces = []
	for group in faceGroups:
		linkedFaces.add(frozenset(group))
		#linkedFaces.append(group)
		
	# create a lookup for the proper key in linkedFaces
	linkedFacesLookup = {}
	for faceGroup in linkedFaces:
		for face in faceGroup:
			linkedFacesLookup[face.getHash()] = faceGroup
		
	return res, linkedFaces, linkedFacesLookup

def getFilteredEdgesSTL(t):

	# in step models, some models will have extranious edges 
	# in the areas between faces, or along cylinders.
	# this function removes those edges.
	
	# NOTE: THIS FUNCTION COULD (AND SHOULD)
	# BE EXPANDED TO FUSE FACES PROPERLY SO GETPATHS2 WORKS 
	# AND TO ALSO FUSE STRAIGHT LINES.

	if type(t) != OCC.Extend.TopologyUtils.TopologyExplorer:
		print("getFilteredEdges REQUIRES an object of type OCC.Extend.TopologyUtils.TopologyExplorer")
		print("not a " + str(type(t)))
		exit(1)
	
	# map. key is the hash of the EDGE. 
	# value is array of format (continuity, face1, face2) \
	# first element of the array is a INT indicating its continuity.
	# https://dev.opencascade.org/doc/refman/html/_geom_abs___shape_8hxx.html#a943632453b69386bece6c091156b1ed5
	
	faceEdgesContinuity = {}
	
	# configure data 
	for face in t.faces():
		# another topology explorer here may be faster, do some testing
		
		wrappedFace = TopoDS_Wrapper(face)
		
		for edge in t.edges_from_face(face):
		
			wrappedEdge = TopoDS_Wrapper(edge)
		
			if wrappedEdge not in faceEdgesContinuity:
				faceEdgesContinuity[wrappedEdge] = [-1]
				
			faceEdgesContinuity[wrappedEdge].append(wrappedFace)
	
	# set continuity 
	bt = BRep_Tool()
	for edge in faceEdgesContinuity.keys():
	
		if len(faceEdgesContinuity[edge]) == 2:
			# only one face had this edge
			continue
	
		if len(faceEdgesContinuity[edge]) != 3:
			print("somehow, an edge had more than 2 faces bordering it? it had " + str(len(faceEdgesContinuity[edge])))			
			exit(1)
			
		face1, face2 = faceEdgesContinuity[edge][1:]	
		
		tempRes = bt.HasContinuity(edge.get(), face1.get(), face2.get())
		if tempRes:
			faceEdgesContinuity[edge][0] = bt.Continuity(edge.get(), face1.get(), face2.get())
	
	
	# for the alternative path gen 
	# if faces have continuity, link them together
	# record what faces have continuity
	# key is the face, val is the array of faces that it has continuity with
	graph = nx.Graph()
	
	res = []
	for k, v in faceEdgesContinuity.items():
	
		if v[0] == 0 and len(v) == 3:
			#k.addFaces(v[1:])
			#k.addFaces([v[1], v[2]])
			k.addFaceHashes([v[1].__hash__(), v[2].__hash__()])
			res.append(k)
			graph.add_node(v[1])
			graph.add_node(v[2])
			#graph.add_node(v[1].__hash__())
			#graph.add_node(v[2].__hash__())
		elif v[0] == 1 and len(v) == 3: # len(v[1:]) == 2

			# add node for each and an edge
			for face in v[1:]:
				if face not in graph.nodes:
					graph.add_node(face)
					#graph.add_node(face.__hash__())
			
			graph.add_edge(v[1], v[2])			
			#graph.add_edge(v[1].__hash__(), v[2].__hash__())
			
			
	# get connected face groups
	
	faceGroups = list(nx.connected_components(graph))
	
	linkedFaces = set()
	#linkedFaces = []
	for group in faceGroups:
		linkedFaces.add(frozenset(group))
		#linkedFaces.append(group)
		
	# create a lookup for the proper key in linkedFaces
	linkedFacesLookup = {}
	for faceGroup in linkedFaces:
		for face in faceGroup:
			linkedFacesLookup[face.getHash()] = faceGroup
	
	newRes = []
	
	hashToFace = {}
	
	for face in t.faces():
		tempFace = TopoDS_Wrapper(face)
		hashToFace[tempFace.getHash()] = tempFace
	
	colors = loadColors()
	colorIndex = 0
	
	for edge in res:
		
		for p in np.linspace(0, 1, 10):
			
			normals = getNormalVector(edge, hashToFace, returnBoth = True, lineProgress = p) 
			v1_u = normals[0] / np.linalg.norm(normals[0])
			v2_u = normals[1] / np.linalg.norm(normals[1])

			angle = (180 / np.pi) * abs(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

			# sometimes, weirdly enough, the normals between 2 faces are flipped.
			# no clue why, but 2 checks are needed now
			#if angle < 45 or angle > 165:
			if angle < 45 or angle > 135:
			#if angle < 75 or angle > 105:
				break

		else:
			
			"""
			colorIndex += 1
			
			curve = BRepAdaptor_Curve(edge.get())
			
			param = 0.5
			normal0, normal1 = getNormalVector(edge, hashToFace, returnBoth = True, lineProgress = param) 
			point = curve.Value(param * (edge.endParameter - edge.startParameter)).Coord()
			
			v1_u = normal0 / np.linalg.norm(normal0)
			v2_u = normal1 / np.linalg.norm(normal1)
			
			angle = (180 / np.pi) * abs(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
			
			#if angle < 100:
			#	continue
			
			#print(angle)
			
			normalPoint0 = np.array(point) + (2 * np.array(normal0))
			normalPoint1 = np.array(point) + (2 * np.array(normal1))
			
			#display.DisplayShape(gp_Pnt(*normalPoint0), color=colors[colorIndex % len(colors)])
			#display.DisplayShape(gp_Pnt(*normalPoint1), color=colors[colorIndex % len(colors)])
			
			display.DisplayShape(edge.get(), color=colors[colorIndex % len(colors)])
			"""
			
			newRes.append(edge)
			
	
	return newRes, linkedFaces, linkedFacesLookup

class MakePickleable:
	# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function

	# IMPORTANT NOTE 
	# when in threading, the hashes for edges are DIFFERENT 

	@staticmethod
	def threadedCollision(jobQueue, returnQueue, shouldStop):

		# fix: overall, collision detection neeeds to 
		# account for the proper size of the tool 
		# account for the surface that the object is on (just do a z check for this one, maybe?)
		# also this should be rewritten, entirely.
		
		tolerance = 0.1
		deflection = 1e2
		
		# dumb code
		# sending the model over direct pickleing was, incredibly slow. like in the minutes 
		# while weird, this is better
		with open("tempSolid.pickle", "rb") as f:
			solid = pickle.load(f)
	
		testMesh = BRepMesh_IncrementalMesh(solid, deflection)
		testMesh.Perform()
	
		detector = BRepClass3d_SolidClassifier(solid)

		# currently, these functions are simple in nature. 
		# they should be made more complex to properly account for the shape and angle of the tool
		curveFunctions = [
		lambda p : (p[0], p[1], p[2] + dif),
		lambda p : (p[0], p[1], p[2] + dif + 1),
		
		
		#] + [ lambda p : (p[0], p[1], p[2] + dif * f + 1) for f in np.arange(0, 1, 0.1) ] + [

		#MakePicklea#ble.testCollisionUp,
		
		#lambda p : (p[0], p[1] + dif + 1, p[2]),
		#lambda p : (p[0], p[1] - dif - 1, p[2]),
		
		#lambda p : (p[0] + dif + 1, p[1], p[2]),
		#lambda p : (p[0] - dif - 1, p[1], p[2]),
		]

		while not shouldStop.is_set():
			
			try:
				data = jobQueue.get(timeout = 0.5)
			except queue.Empty:
				time.sleep(0.001)
				continue

			edgeWrapper = data
			
			curve = BRepAdaptor_Curve(edgeWrapper.get())
			
			start, end = curve.FirstParameter(), curve.LastParameter()
		
			current = start
	
			edgeData = []
			
			# dont ask
			# decimals being long messed things up, for some reason
			# FIX: this needs to be looked at more. maybe use round???
			# ideally just get all vertexes from the edges directly
			dif = (end - start) / 10
			#dif = (end - start) / 50
			difPrecision = 1000
			dif = math.ceil(dif * difPrecision) / difPrecision
			while current < end:
			
				next = current + dif
				
				# lookahead, extend this one if within a certain amount
				if end - next < 1/1000:
					next = end

				#p = curve.Value(current)
				#p.SetZ(p.Z() + dif + 1)
				
				point = curve.Value(current)
				originalCoords = (point.X(), point.Y(), point.Z())
			
				for curveFunc in curveFunctions:
	
					point.SetCoord(*curveFunc(originalCoords))
					
					if point.Z() < -70:
						# THIS IS VERY BADDDDD
						continue
					
					if point.Z() < 0:
						# THIS IS EVEN WORSE
						continue
										
						
					detector.Perform(point, tolerance)
					
					if detector.State() == 1: 
					
						# valid edge.
						edgeData.append([True, [current, next]])
						break
				else:
					# invalid edge
					# we could just not append here and maybe save some cycles 
					# but for debug it is nice
					
				
					edgeData.append([False, [current, next]])
					
					pass
				
				current = next
				
			returnQueue.put([data, edgeData])

		pass

#@jit(target_backend='cuda')
#@jit(target_backend='cuda', forceobj=True, nogil=True, parallel=True)
@jit(target_backend='cuda', forceobj=True)
def gpuCollision(edges, solid, res):
	
	# ideally this function would be in the makepickleable, 
	# but its possible that thats with the jit
	
	tolerance = 0.1
	deflection = 1e2
	
	#testMesh = BRepMesh_IncrementalMesh(solid, deflection)
	#testMesh.Perform()
	#detector = BRepClass3d_SolidClassifier(solid)
	
	#print("bruh", len(edges))

	#for edgeWrapper in edges:
	for i in range(0, len(edges)):
	
		#testMesh = BRepMesh_IncrementalMesh(solid, deflection)
		#testMesh.Perform()
		#detector = BRepClass3d_SolidClassifier(solid)
	
	
		edgeWrapper = edges[i]
		
		curve = BRepAdaptor_Curve(edgeWrapper.get())
		
		start, end = curve.FirstParameter(), curve.LastParameter()
	
		current = start

		edgeData = []
		
		# dont ask
		# decimals being long messed things up, for some reason
		# FIX: this needs to be looked at more. maybe use round???
		# ideally just get all vertexes from the edges directly
		dif = (end - start) / 10
		#dif = (end - start) / 50
		difPrecision = 1000
		dif = math.ceil(dif * difPrecision) / difPrecision
		
		curveFunctions = (
			(0, 0, dif),
			(0, 0, dif + 1),
		)


		while current < end:
		
			next = current + dif
			
			# lookahead, extend this one if within a certain amount
			if end - next < 1/1000:
				next = end
				
			point = curve.Value(current)
			originalCoords = (point.X(), point.Y(), point.Z())
		
			# break is bugged, and kills ALL LOOPS.
			validFound = False
		
			for curveFunc in curveFunctions:

				if validFound:
					continue
					
				# THIS IS SHIT 
				if originalCoords[2] < 31:
					continue

				#point.SetCoord(*curveFunc(originalCoords, dif))
				point.SetCoord(
				originalCoords[0] + curveFunc[0],
				originalCoords[1] + curveFunc[1],
				originalCoords[2] + curveFunc[2]
				)
				
				if point.Z() < -70:
					# THIS IS VERY BADDDDD
					continue
				
				# HEY HEY DUMBASS, THIS IS AND SHOULD BE AT 0 BUT ISNT BC STUPID SHIT.
				#if point.Z() < 0:
				if point.Z() < 31.99:
					# THIS IS EVEN WORSE
					continue
					
				# THIS IS BAD AND SHOULD NOT BE DONE 
				# THE DETECTOR IS NOT BEING SHARED 
				# PROPERLY BETWEEN THREADS
				
				#detector.Perform(point, tolerance)
				
				#if detector.State() == 1: 
				if True:
					# valid edge.
					edgeData.append([True, [current, next]])
					validFound = True
					# THIS BREAK WILL BREAK BOTH LOOPS FOR SOME REASON 
					# DONT USE BREAK
					#break
			else:
				
				if not validFound:
					edgeData.append([False, [current, next]])
				
				pass
			
			current = next
		
		
		res[i] = [edgeWrapper, edgeData]
		#res[i] = [str(i) + "bruh"]
		#print(i)
		#res[i] = i
		
	pass
	
def getCollision(edges, solid):

	# eventually, the specifc angle that the collision works on 
	# will NEED to be stored in the edge for later processing

	# i have 0 clue what a good tolerance is 
	tolerance = 0.1
	deflection = 1e-2
	
	edgeCollisionPool = PoolQueue(MakePickleable.threadedCollision, cpuPercent = 0.75)
	
	# this is insanely dumb 
	# but for some reason, passing the pickle through directly to 
	# the processes takes a very long time
	# while this doesnt.
	with open("tempSolid.pickle", "wb") as f:
		pickle.dump(solid, f)

	edgeCollisionPool.start()

	numEdges = len(edges)
	
	for i, edgeWrapper in enumerate(edges):	
		edgeCollisionPool.send(edgeWrapper)
		print("working: {:6.2f}% done".format(100 * i / (numEdges - 1)), end="\r")
	
	print("")
		
	res = edgeCollisionPool.join()
	

	# merge consecutive segments with the save validity
	# this COULD be done in the threading! but doesnt take up enough time to warrant it
	validEdges = []
	
	for edgeWrapper, edgeData in res.items():
		
		newEdgeData = [edgeData[0]]
		
		for testEdgeData in edgeData[1:]:
			
			if testEdgeData[0] != newEdgeData[-1][0]:
				# if this edge segment has a different validiy than the most recent one
				# update the last segments end point to this segments start, and add this segment to the new data
				# this code is trash, make it more readable
				newEdgeData[-1][1][1] = testEdgeData[1][0]
				newEdgeData.append(testEdgeData)
		
		# set the end of the newedgedata to the proper end
		newEdgeData[-1][1][1] = edgeData[-1][1][1]
			
		for newEdge in newEdgeData:
			if newEdge[0]:
				validEdges.append(trimEdge(edgeWrapper, newEdge[1][0], newEdge[1][1]))

	return validEdges

def getCollisionGPU(edges, solid):

	print("using gpu collision, going fast, no progress bar tho")
	print("actual collision detection is also not as good as the cpu version")

	# suppress numba debug output
	import logging
	numba_logger = logging.getLogger('numba')
	numba_logger.setLevel(logging.WARNING)
	
	# i have 0 clue what a good tolerance is 
	#tolerance = 0.1
	#deflection = 1e-2


	tempRes = [None] * len(edges)
	#res = list(range(0, len(edges)))
	#res = np.zeros(len(edges), dtype = np.float64)
	
	#res = numbaDict.empty(
	#	key_type = types.uint32,
	#	value_type = _TopoDS_EdgeWrapper,
	#)
	
	#MakePickleable.gpuCollision(edges, solid, res)
	gpuCollision(edges, solid, tempRes)
	
	
	res = {}
	
	for d in tempRes:
		res[d[0]] = d[1]
	

	# merge consecutive segments with the save validity
	# this COULD be done in the threading! but doesnt take up enough time to warrant it
	validEdges = []
	
	
	for edgeWrapper, edgeData in res.items():
		
		newEdgeData = [edgeData[0]]
		
		for testEdgeData in edgeData[1:]:
			
			if testEdgeData[0] != newEdgeData[-1][0]:
				# if this edge segment has a different validiy than the most recent one
				# update the last segments end point to this segments start, and add this segment to the new data
				# this code is trash, make it more readable
				newEdgeData[-1][1][1] = testEdgeData[1][0]
				newEdgeData.append(testEdgeData)
		
		# set the end of the newedgedata to the proper end
		newEdgeData[-1][1][1] = edgeData[-1][1][1]
			
		for newEdge in newEdgeData:
			if newEdge[0]:
				validEdges.append(trimEdge(edgeWrapper, newEdge[1][0], newEdge[1][1]))
	
	
	return validEdges

def testValidAppend(graph, existingPoint, testEdge, *, _ignoreParallel = False):

	# tests if an edge can be appended
	# an edge can be appended if the edge it is trying to be added to does not already have 
	# 2 valid edges, meaning it is connected to a path, and 
	# is parallel to the edge at the point.

	if len(graph.nodes[existingPoint]["edgeData"]) != 1:
		# if we are here, then we are assuming that the old point has already found a valid path. 
		return False
	else:
		# see if the new edge would be parallel to what it is being appended to
		
		if _ignoreParallel:
			return True
		
		otherEdgeWrapper = graph.nodes[existingPoint]["edgeData"][0]
		
		return testEdge.isParallel(otherEdgeWrapper, existingPoint)

	pass

def getPaths(validEdges, *, _recurseIndex = 0, _ignoreParallel = False):
	
	# generate paths. 
	# this function takes parallel lines 
	# into account, and seperates non parallel lines

	if _recurseIndex > 50:
		print("getPaths seemed to get into an infinite loop. this is not ideal")
		exit(1)
	
	if len(validEdges) == 0:
		return []
	
	unsolvedEdges = set()
	
	# convert the 3d graph(model) into a 2d graph in order to get groups of connected edges and create paths
	graph = nx.Graph()

	for edgeWrapper in validEdges:

		if edgeWrapper.isClosed():
			# curve is closed
			tempPoint = edgeWrapper.getStartPoint()
			graph.add_node(tempPoint)
			graph.nodes[tempPoint]["edgeData"] = [edgeWrapper] 
			continue

		newPoints = [edgeWrapper.getStartPoint(), edgeWrapper.getEndPoint()]
		existingPointUsed = [False, False]
		for i in range(0, len(newPoints)):
			# this code could be better. making sure that the starts and ends of the edges 
			# are the same would be better than this solution, which currently involves rounding.
			if newPoints[i] not in graph.nodes:
				graph.add_node(newPoints[i])
				graph.nodes[newPoints[i]]["edgeData"] = []
			else:
				existingPointUsed[i] = True
		
		
		if existingPointUsed == [False, False]:
			# both points are new, just add the points and add the edge
			
			for i in range(0, 2):
				graph.nodes[newPoints[i]]["edgeData"].append(edgeWrapper)
			graph.add_edge(*newPoints)
			
		elif existingPointUsed == [True, False] or existingPointUsed == [False, True]:
			# one new point, check if it is parallel to the poing being added on!
			
			if existingPointUsed == [False, True]:
				# swap points such that the existing one is always the first 
				newPoints[0], newPoints[1] = newPoints[1], newPoints[0]
				existingPointUsed = [True, False]
				
			if testValidAppend(graph, newPoints[0], edgeWrapper, _ignoreParallel = _ignoreParallel):
				graph.nodes[newPoints[0]]["edgeData"].append(edgeWrapper)
				graph.nodes[newPoints[1]]["edgeData"].append(edgeWrapper)
				graph.add_edge(*newPoints)
			else:
				graph.nodes[newPoints[1]]["edgeData"].append(edgeWrapper)

		else: # existingPointUsed == [True, True]
			# closing a shape, both points exist, check BOTH

			testAppends = [testValidAppend(graph, newPoints[0], edgeWrapper, _ignoreParallel = _ignoreParallel), testValidAppend(graph, newPoints[1], edgeWrapper, _ignoreParallel = _ignoreParallel)]
			
			if testAppends == [False, False]:
				# certain edges cannot be solved within the current bounds of this program within one loop
				# keep track of them, and then recurse
				unsolvedEdges.add(edgeWrapper)
			elif testAppends == [True, False] or testAppends == [False, True]:
				if testAppends == [False, True]:
					newPoints[0], newPoints[1] = newPoints[1], newPoints[0]
					testAppends = [True, False]
				graph.nodes[newPoints[0]]["edgeData"].append(edgeWrapper)
			else: # testAppends == [True, True]
				graph.nodes[newPoints[0]]["edgeData"].append(edgeWrapper)
				graph.nodes[newPoints[1]]["edgeData"].append(edgeWrapper)
				graph.add_edge(*newPoints)

	# get groups of connected nodes from the graph, each of which is a path
	graphGroups = list(nx.connected_components(graph))
	
	resultPaths = []
	
	for i, groupedNodes in enumerate(graphGroups):
		
		groupedEdges = set()
	
		for node in groupedNodes:
			for edge in graph.nodes[node]["edgeData"]:
				groupedEdges.add(edge)
		
		"""
		topoGroupedEdges = TopTools_ListOfShape()
		for e in groupedEdges:
			topoGroupedEdges.Append(e.get())
		
		outputWire = BRepBuilderAPI_MakeWire()
		outputWire.Add(topoGroupedEdges)
		resultPaths.append(outputWire.Wire())
		"""

		resultPaths.append(groupedEdges)
		
	if len(unsolvedEdges) != 0:
		# very very very weird code ahead.
		# basically, sometimes when a edge has 2 existing points, but no valid append(due to not being parallel)
		# things are just, bad.
		# by recording those edges, and then recursing on them 
		# to generate unique paths from just those edges, until none are left
		# however, this has potential to maybe inf loop
		# im adding a index to prevent this and at least give an error 
		# but this should maybe be looked into more
		resultPaths.extend(getPaths(unsolvedEdges, _recurseIndex=_recurseIndex+1, _ignoreParallel = _ignoreParallel))
	
	return resultPaths

def getPaths2(validEdges, linkedFaces, linkedFacesLookup):

	# generate paths. 
	# this function does not take parallel lines into account

	
	# key is face, and the value is an array of the edges on that face
	edgeFaces = {}

	for faceGroup in linkedFaces:
		edgeFaces[faceGroup] = set()
		
	for edge in validEdges:
		for faceHash in edge.getFaceHashes():
			edgeFaces[linkedFacesLookup[faceHash]].add(edge)
	
	#edgeFaces = sorted(edgeFaces.items(), key=lambda item: len(item[1]), reverse=True)
	
	# this counts not just what would be in the path but the length of all the edges in this.
	# this may not be ideal, but is better than just quantity
	# also this is weirdly enough my first time using map, i would 
	# normally just go through and make a whole for loop for this but i want to / need to one line it
	edgeFacesList = sorted(edgeFaces.items(), key=lambda item: sum(list(map(lambda obj: obj.length, item[1]))), reverse=True)
	
	usedEdges = set()
	
	resultPaths = []
	
	#for face, edges in edgeFacesList:
	while True:
		
		if len(edgeFaces) == 0:
			break
	
		edgeFacesList = sorted(edgeFaces.items(), key=lambda item: sum(list(map(lambda obj: obj.length, item[1]))), reverse=True)
		face, edges = edgeFacesList[0]
		
		del edgeFaces[face]
	
		edges = edges - usedEdges
		
		if len(edges) == 0:
			continue
	
		paths = getPaths(edges, _ignoreParallel = True)
		
		resultPaths.extend(paths)
		
		usedEdges.update(edges)

	return resultPaths

def getOffsetPath(pathData, hashToFace):

	# gens a path with a certain dist to each edge, with normals to said edge 
	# does not do anything collision related tho
	# purely for a hypothetical prox sensor

	# could use wires here, but it causes some issues (edge ordering, accessing per face edge data)

	offsetPathData = []

	tool = BRep_Tool()	
	
	for path in pathData:
	
		offsetPath = []
		
		for edge in path:
		
			# this param NEEDS TO BE FOUND DYNAMICALLY
			# EXAMPLE, IF OFFSET IS GREATER THAN A CURVES RADIUS, BAD THINGS OCCUR
			offsetDistance = 5
			#offsetDistance = 3
			
			wire = TopoDS_Wrapper(BRepBuilderAPI_MakeWire(edge.get()).Wire())
		
			startPoint = edge.startPoint
			startVector = edge.startVector
			offsetVector, angle = getOffsetVector(edge, hashToFace)
			
			goalPoint = np.add(startPoint, np.multiply(offsetVector, offsetDistance))
			
			# i have tried over a dozen ways of doing this. this is, sadly
			# the best/only method i could get to work
			# more importantly, this MAINTAINS PRECISION.
			
			circle = gp_Circ(gp_Ax2(gp_Pnt(*startPoint), gp_Dir(*startVector)), offsetDistance)
			
			circleEdge = BRepBuilderAPI_MakeEdge(circle).Edge()
			
			pipeMaker = BRepOffsetAPI_MakePipe(wire.get(), circleEdge)
		
			pipe = pipeMaker.Shape()

			#display.DisplayShape(pipe, color=c, transparency=0.9)
			#display.DisplayShape(pipe, color=c)
			
			# most likely inefficent. there MUST be a better way of doing this
			pipeExplorer = TopExp_Explorer(pipe, TopAbs_FACE)
		
			faces = []
			while pipeExplorer.More():
				faces.append(pipeExplorer.Value())
				pipeExplorer.Next()
			
			if len(faces) != 1:
				print("a single edge has multiple faces in it. this should not happen")
				exit(1)
			
			face = faces[0]
			
			faceHandle = tool.Surface(face)
			
			shapeAnalyzer = ShapeAnalysis_Surface(faceHandle)
			
			uClosed, vClosed = shapeAnalyzer.IsUClosed(), shapeAnalyzer.IsVClosed()
			
			if [uClosed, vClosed] == [0, 0]:
				print("neither uv on the surface was closed. This should not happen")
				exit(1)
			elif [uClosed, vClosed] == [0, 1]:
			
				# no clue what this precision should be
				u, v = shapeAnalyzer.ValueOfUV(gp_Pnt(*goalPoint), 0.1).Coord()
				
				curve = shapeAnalyzer.VIso(v)
			elif [uClosed, vClosed] == [1, 0]:
				
				# no clue what this precision should be
				u, v = shapeAnalyzer.ValueOfUV(gp_Pnt(*goalPoint), 0.1).Coord()
				
				curve = shapeAnalyzer.UIso(u)
			else: # [uClosed, vClosed] == [1, 1]:
				
				# in order to figure out which way to go 
				# invert the goal point calc, and then see which param doesnt change 
				# this way we figure out which of the uvs is actually correct
				
				# this area NEEDS more testing
				
				u, v = shapeAnalyzer.ValueOfUV(gp_Pnt(*goalPoint), 0.1).Coord()
				
				goalPoint2 = np.add(startPoint, np.multiply(np.negative(offsetVector), offsetDistance))
				
				u2, v2 = shapeAnalyzer.ValueOfUV(gp_Pnt(*goalPoint2), 0.1).Coord()
				
				# half a rotation is pi. the closest 
				# axis to pi is the one we should NOT take. 
				
				uDif = abs(abs(u - u2) - np.pi)
				vDif = abs(abs(v - v2) - np.pi)
				
				
				if abs(uDif - vDif) < 1:
					# there was not a large enough difference between udif and vdif to discern which way to go")
					# this should never happen. it may, however, if offsetdis is very low/ similar to the radius of the current edge")
					# GENERATE BOTH CURVES, CHECK THEIR VECTORS")
					# how expensive even is curve gen, this will not save much time tbh but still 
					
					print("WRITE THIS IDIOT")
					exit(1)
					
				if uDif > vDif:
					curve = shapeAnalyzer.VIso(v)
		
				else: # uDif < vDif
					curve = shapeAnalyzer.UIso(u)
		
				#display.DisplayShape(face, color=red, transparency=0.95)
		
			testEdge = BRepBuilderAPI_MakeEdge(curve).Edge()
			
			#display.DisplayShape(wire.get(), color=green)
			#display.DisplayShape(face, color=c, transparency=0.95)
			#display.DisplayShape(testEdge, color=white)
		
			#display.DisplayShape(wire.get(), color=c)
			#display.DisplayShape(testEdge, color=c)
			
			#offsetPath.append(_TopoDS_EdgeWrapper(testEdge))
			# display.DisplayShape(face, color=c, transparency=0.95)
			#display.DisplayShape(testEdge, color=c)
			
			
			#display.DisplayShape(testEdge, color=red)
	
			
			#offsetPath.append([edge, _TopoDS_EdgeWrapper(testEdge), offsetVector])
			#offsetPath.append([edge, _TopoDS_EdgeWrapper(testEdge), angle])
			
		
			offsetPath.append([edge, _TopoDS_EdgeWrapper(testEdge)])
		
		offsetPathData.append(offsetPath)

	return offsetPathData 

def getRapid3(offsetPathData, hashToFace):

	# todo, keep isparallel at 5, but then once we get to (maybe this function honestly)
	# we need to identify the parts of the paths that can be combined
	# ALSO
	# the start and end points of some lines are weirdly being off by ~1?
	# unsure about what thats about
	# also, look more into pyvista's extract edges function 
	# see if we can easily reconstruct a solid from that
	# also like, combining the parts should be maybe done earlier to
	# reduce the amount of work all the other funcs have to do
	# hell now i dont even need the entire offset function, other than
	# maybe for tool angles?
	# as in like we can just adjust the toolframe, and all we need now are the,
	# well we still need offset angles but dont need offset lines
	
	# using gpu for collision makes my worries about edge quantity slowing stuff 
	# earlier on much much better. 
	# even with a thousand edges, 4 seconds.
	# and most of that is probs spent copying memory.

	# wow.
	# axialy mounting the tool really went and made this
	# WHOLE THING USELESS
	# god.

	# initial sanity check 
	for path in offsetPathData:
		for edgePair in path:
		
			edge, offsetEdge = edgePair
		
			s = edge.startParameter == offsetEdge.startParameter
			e = edge.endParameter == offsetEdge.endParameter
			
			if not s or not e:
				print("start and end parameters of edge were different. this is very bad")
				exit(1)
	
	#calibPoint = [1000, 0, 300]
	#calibPoint = [1000, 0, 300]
	calibPoint = [900, 0, 550]
	#calibPoint = [1200, 0, 0]
	calibPoint = np.array(calibPoint)
	
	
	# THIS IS BAD AND NEEDS A PER LINETYPE FIX
	pointsPerLine = 10
	
	f = open("rapid.txt", "w")
	f.write("MODULE MainModule\n\n")
	f.write("\tgoHome;\n")
	
	colorIndex = 0
	globalColorIndex = 0
	colors = loadColors()
	
	for originalPath in offsetPathData:

		#if len(path) < 10:
		#if len(path) < 50:
			# this is bad!
		#	continue
		
		lengthTotal = 0
		for edgePair in originalPath:
			lengthTotal += edgePair[0].length
			
		#if lengthTotal < 20:
		if lengthTotal < 25:
			continue
		
		# this DOES introduce some imprecision!!!
		orderedPath = orderPath(originalPath)
				
		# this should combine parallel lines into large edges 
		# and also (hopefully) create arcs/circles/whatever
		#segmentedPath = segmentPath(orderedPath)
		
		f.write("\t!starting new path\n")
		
		globalColorIndex += 1
		colorIndex += 1
		
		#for edgePair in originalPath:
		#	display.DisplayShape(edgePair[0].get(), color=colors[colorIndex % len(colors)])
		#	display.DisplayShape(edgePair[1].get(), color=red)
				
		subPaths = combineSTLPath(orderedPath)
	
		for path in subPaths:
			
			#f.write("\t\t!starting new subpath\n")
		
			lineType = path[0]
			path = path[1]
		
			colorIndex += 1
			
			# this is just meant to help filter out the inacurate smaller 
			# circles, and is TRASH
			if len(path) < 3:
				lineType = False
			
			c = white if lineType else red
			
			#for edgePair in path:
			#	display.DisplayShape(edgePair[0].get(), color=colors[colorIndex % len(colors)])
			#	display.DisplayShape(edgePair[1].get(), color=c)
			
			if lineType == False: # line
			
				for edgePair in path:
					display.DisplayShape(edgePair[0].get(), color=colors[globalColorIndex % len(colors)])
					#display.DisplayShape(edgePair[0].get(), color=colors[colorIndex % len(colors)])
					#display.DisplayShape(edgePair[1].get(), color=red)
				
				startEdge, startOffsetEdge = path[0]
				endEdge, endOffsetEdge = path[-1]
				
				startOffsetPoint, startToolAngles = getToolTarget(startEdge, startOffsetEdge, startEdge.startParameter)
				endOffsetPoint, endToolAngles = getToolTarget(endEdge, endOffsetEdge, endEdge.endParameter)
				
				# distrust
				startCurve = BRepAdaptor_Curve(startEdge.get())
				endCurve = BRepAdaptor_Curve(endEdge.get())
				
				startPoint = np.array(startCurve.Value(startCurve.FirstParameter()).Coord())
				endPoint = np.array(endCurve.Value(endCurve.LastParameter()).Coord())
				
				#f.write("\tres := doLine([{:f},{:f},{:f},{:f},{:f}], [{:f},{:f},{:f},{:f},{:f}]);\n".format(
				#*startOffsetPoint, *startToolAngles, *endOffsetPoint, *endToolAngles))
				
				f.write("\t\tres := doLine([{:f},{:f},{:f},{:f},{:f}], [{:f},{:f},{:f},{:f},{:f}]);\n".format(
				*startPoint, *startToolAngles, *endPoint, *endToolAngles))
			
			else: # circle 
				#print("skipping circle")
				
				for edgePair in path:
					display.DisplayShape(edgePair[0].get(), color=colors[globalColorIndex % len(colors)])
					#display.DisplayShape(edgePair[0].get(), color=colors[colorIndex % len(colors)])
					#display.DisplayShape(edgePair[1].get(), color=white)
				
				# why am i commenting code more now just bc like,,, idek 
				# gods
				# format, edgepair, param
				
				for i in range(0, len(path)):
					path[i][0].reCalculate()
					path[i][1].reCalculate()
				
				start = [path[0], path[0][0].startParameter]
				end = [path[-1], path[-1][0].endParameter]
				
				if len(path) % 2 == 1:
					# path has odd num of edges, take mid of mid edge
					p = path[len(path) // 2]
					mid = [p, (p[0].endParameter - p[0].startParameter) / 2]
				else:
					# even number of edges, return start of the mid edge
					p = path[len(path) // 2]
					mid = [p, p[0].startParameter]
					
		
				points = [None] * 3
				
				for i, idk in enumerate([start, mid, end]):
					
					edgePair, param = idk
					edge, offsetEdge = edgePair
					
					curve = BRepAdaptor_Curve(edge.get())
					points[i] = np.array(curve.Value(param).Coord())
					
					display.DisplayShape(gp_Pnt(*points[i]), color=white)
					
						
				# i really, REALLY should just write a getpointatparam 
				# func in topods edgewrap 
				# but ill feel bad for all the previous things that 
				# wont use it and i dont want to go back
				
				# https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
				
				a, b, c = points
				
				ba = a - b
				bc = c - b

				cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
				angle = 180 / np.pi * np.arccos(cosine_angle)
				
				if angle > 260:
					# should be 270, but not risking it
					print("movec has a 270 angle limit!!!!!")
					print("write the circle division code idiot.")
					exit(1)
					
				
				startOffsetPoint, startToolAngles = getToolTarget(start[0][0], start[0][1], start[1])
				midOffsetPoint, midToolAngles = getToolTarget(mid[0][0], mid[0][1], mid[1])
				endOffsetPoint, endToolAngles = getToolTarget(end[0][0], end[0][1], end[1])
				
				f.write("\t\tres := doCircle([{:f},{:f},{:f},{:f},{:f}], [{:f},{:f},{:f},{:f},{:f}], [{:f},{:f},{:f},{:f},{:f}]);\n".format(
				*points[0], *startToolAngles, *points[1], *midToolAngles, *points[2], *endToolAngles))
			
	
		f.write("\tgoHome;\n")
		f.write("\n")
		
	
		
	f.write("ENDMODULE\n")
	f.close()



	pass

# 

def finalTest():
	
	#filename = "Rotor_Head_V1.0.stp"
	#filename = "Rotor_Head_V1.2.stp"
	#filename = "complexTest.stp"
	#filename = "nist_ctc_01_asme1_rd.stp"
	#filename = "nist_ctc_02_asme1_rc.stp"
	#filename = "nist_ctc_04_asme1_rd.stp"
	#filename = "top_plate_work_piece.stp"
	filename = "scan0.stp"
	
	print("-----")
	
	print("loading model " + filename)
	startTime = time.perf_counter()
	#t = loadFile(filename)
	t = pointCloudToStep()
	#return
	solid = list(t.solids())[0]
	endTime = time.perf_counter()
	print("loading model took {:.02f} seconds and has {:d} edges".format(endTime - startTime, t.number_of_edges()))
	print("currently using {:s} of ram ".format(getCurrentRamUsage()))
	
	#display.DisplayShape(solid)
	#return
	
	print("-----")	

	print("getting filtered edges")
	startTime = time.perf_counter()
	#edges, linkedFaces, linkedFacesLookup = getFilteredEdges(t)
	edges, linkedFaces, linkedFacesLookup = getFilteredEdgesSTL(t)
	endTime = time.perf_counter()
	print("filtered edges took {:.02f} seconds".format(endTime - startTime))
	print("{:d} valid filtered edges found".format(len(edges)))
	print("currently using {:s} of ram ".format(getCurrentRamUsage()))	
	
	#display.default_drawer.SetFaceBoundaryDraw(False)
	#display.DisplayShape(solid, color=rgb_color(0.9, 0.9, 0.9), transparency=0.95)
	#display.default_drawer.SetFaceBoundaryDraw(True)
	
	#display.DisplayShape(solid)
	#display.DisplayShape(solid, transparency=0.50)

	hashToFace = {}
	
	for face in t.faces():
		tempFace = TopoDS_Wrapper(face)
		hashToFace[tempFace.getHash()] = tempFace
	
	
	#for edge in edges:
	#for edge in t.edges():
		#display.DisplayShape(edge.get())
		#display.DisplayShape(edge)
	#return
	
	#print("REMOVE THIS")
	#edges = edges[:100]
	
	print("-----")

	print("getting collision on {:d} edges".format(len(edges)))
	startTime = time.perf_counter()
	#validEdges = getCollision(edges, solid)
	validEdges = getCollisionGPU(edges, solid)
	endTime = time.perf_counter()
	print("collision took {:.02f} seconds".format(endTime - startTime))
	print(str(len(validEdges)) + " valid edge segments found")
	print("currently using {:s} of ram ".format(getCurrentRamUsage()))

	#for edge in validEdges:
	#	display.DisplayShape(edge.get())
	#return
	
	print("-----")
	
	print("getting paths on {:d} segments".format(len(validEdges)))
	startTime = time.perf_counter()
	pathData = getPaths(validEdges)
	#pathData = getPaths2(validEdges, linkedFaces, linkedFacesLookup)
	endTime = time.perf_counter()
	print("paths took {:.02f} seconds".format(endTime - startTime))
	print(str(len(pathData)) + " paths found")
	print("currently using {:s} of ram ".format(getCurrentRamUsage()))
	
	"""
	colors = loadColors()
	colorIndex = 0
	for path in pathData:
		colorIndex += 1
		for edge in path:
			display.DisplayShape(edge.get(), color=colors[colorIndex % len(colors)])
	return
	"""
	
	print("-----")
	
	print("getting offset paths on {:d} paths".format(len(pathData)))
	startTime = time.perf_counter()
	offsetPathData = getOffsetPath(pathData, hashToFace)
	endTime = time.perf_counter()
	print("offset paths took {:.02f} seconds".format(endTime - startTime))
	print("currently using {:s} of ram ".format(getCurrentRamUsage()))
		
	print("getting rapid on {:d} paths".format(len(offsetPathData)))
	startTime = time.perf_counter()
	#rapid = getRapid(offsetPathData, hashToFace)
	#rapid = getRapid2(offsetPathData, hashToFace)
	rapid = getRapid3(offsetPathData, hashToFace)
	endTime = time.perf_counter()
	print("rapid took {:.02f} seconds".format(endTime - startTime))
	print("currently using {:s} of ram ".format(getCurrentRamUsage()))
	
	print("-----")
	
		
	display.default_drawer.SetFaceBoundaryDraw(False)
	#display.DisplayShape(solid, color=rgb_color(0.9, 0.9, 0.9), transparency=0.95)
	display.default_drawer.SetFaceBoundaryDraw(True)
	
	colors = loadColors()
	
	#for i, path in enumerate(pathData):
	#	display.DisplayShape(path, color=colors[i % len(colors)])

	#for i, path in enumerate(pathData):
	#	for edge in path:
	#		display.DisplayShape(edge.get(), color=colors[i % len(colors)])
	
	"""
	for i, path in enumerate(offsetPathData):
		for edges in path:
			edge, offsetEdge = edges
			display.DisplayShape(edge.get(), color=colors[i % len(colors)])
	#		#display.DisplayShape(offsetEdge.get(), color=colors[i % len(colors)])
			display.DisplayShape(offsetEdge.get(), color=red)
	#		pass
	"""
	
	print("done processing")

	pass
	
#

if __name__ == "__main__":

	print("initializing display")

	display, start_display, add_menu, add_function_to_menu = init_display(backend_str="qt-pyqt5")
	
	display.EraseAll()
	
	setupControls(display)
	
	display.View.SetBgGradientColors(
		Quantity_Color(Quantity_NOC_BLACK),
		Quantity_Color(Quantity_NOC_BLACK),
		2,
		True)
		
	display.Repaint()
	
	#finalTest()
	
	# for reasons unknown to the gods, the memory profiler 
	# will DUPLICATE CALLS TO THIS FUNCTION
	# if it does not allocate enough memory. 
	# this prevents that from occuring
	# https://github.com/CFMTech/pytest-monitor/issues/26
	# https://github.com/pythonprofilers/memory_profiler/issues/298
	# the function needs to take a certain amount of time to prevent this. 
	# insanely stupid bug
	# either that, or there is max_iterations, but the default behaviour really should not act like this 
	memoryUsage = memory_usage(finalTest, max_iterations=1)

	print("max memory usage was {:s}".format(getCurrentRamUsage(max(memoryUsage), suffix="MiB")))
	
	display.FitAll()
	start_display()	

	print("exiting")


