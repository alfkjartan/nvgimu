""" xyzplot.py
Chaco plot which lets the user choose which of the x, y and z component of a 3D 
time series to plot.
"""
import numpy as np
from chaco.tools.api import PanTool, ZoomTool, DragZoom
from chaco.api import ArrayPlotData, Plot
from enable.component_editor import ComponentEditor
from traits.api import Dict, List, HasTraits, Instance
from traitsui.api import Item, View, CheckListEditor

class XYZChooser(HasTraits):

    plot = Instance(Plot)
    data_names = List(editor=CheckListEditor(values=["x", "y", "z"]))
    traits_view = View(Item('data_names', label="Elements", style="custom"),
                       Item('plot', editor=ComponentEditor(), show_label=False),
                       width=800, height=600, resizable=True,
                       title="XYZ Timeseries")

    colors = {"x": "blue", "y": "green", "z": "red"}
    def __init__(self, times, data):
        self.data_ = data

        # Create the data and the PlotData object
        self.plotdata = ArrayPlotData(time=times, 
                                      xdata=self.data_[:,0],
                                      ydata=self.data_[:,1],
                                      zdata=self.data_[:,2])

        # Create a Plot and associate it with the PlotData
        plot = Plot(self.plotdata)
        
        # Create line plots
        plot.plot(("time", "xdata"), type="line", color="blue", name="x")
        plot.plot(("time", "ydata"), type="line", color="green", name="y")
        plot.plot(("time", "zdata"), type="line", color="red", name="z")
        
        for p in self.colors.keys():
            plot.hideplot(p)

        plot.tools.append(PanTool(plot))
        plot.tools.append(ZoomTool(plot, tool_mode='box', always_on=True, drag_button="right"))
        #plot.tools.append(DragZoom(plot, drag_button="right"))

        self.plot = plot

    def _data_names_changed(self):
        
        for p in self.colors.keys():
            self.plot.hideplot(p)
        
        for p in self.data_names:
            self.plot.showplot(p)

#===============================================================================
# demo object that is used by the demo.py application.
#===============================================================================
if __name__ == "__main__":
    times = np.linspace(0,10, 50)
    data = np.random.random((times.size,3))
    demo=XYZChooser(times, data)
    demo.configure_traits()

