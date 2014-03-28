import math
import itertools
import matplotlib.pyplot as pyplot

class PointFinder:
  """
  callback for matplotlib to store the point  where the user clicks.
      
  Register and use this functor like this:
    
  scatter(xdata, ydata)
  af = PointFinder()
  connect('button_press_event', af)
  ...
  selectedPoints = af.get_selected()

  """

  def __init__(self, xdata, axis=None, xtol=None):
    if xtol is None:
      xtol = (max(xdata) - min(xdata))/100.0
      self.xtol = xtol
    if axis is None:
      self.axis = pyplot.gca()
    else:
      self.axis= axis
    self.drawnAnnotations = {}
    self.points = []
    self.lines = []

  def __call__(self, event):
    #print "Received button event %d" % (event.button,)
    if event.inaxes and event.button==3: # Right button pressed inside plot window
      clickX = event.xdata
      clickY = event.ydata
      if self.axis is None or self.axis==event.inaxes:
        # If close to previous selected point, then remove this
        pointremoved = False
        for [p, l] in itertools.izip(self.points, self.lines):
          if abs(clickX - p) < self.xtol:
            self.points.remove(p)
            self.lines.remove(l)
            pyplot.draw()
            pointremoved = True
        if not pointremoved:
          self.points.append(clickX)
          yl = pyplot.get(self.axis, 'ylim')
          self.lines.append(pyplot.plot([clickX, clickX], yl, 'r'))
          pyplot.draw()

  def get_selected(self):
      return self.points
