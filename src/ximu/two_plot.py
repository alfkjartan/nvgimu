""" Adopted from chacos standard example two_plot.py """

# Major library imports
from numpy import arange
from scipy.special import jn

# Enthought library imports
from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, Group, View

# Chaco imports
from chaco.api import HPlotContainer, ArrayPlotData, Plot
from chaco.tools.api import LineInspector, ZoomTool, PanTool

#===============================================================================
# Attributes to use for the plot view.
size=(750,500)
title="Two Plots"

#===============================================================================
# # Plotwndw class that is used by the plotwndw.py application.
#===============================================================================
class Plotwndw(HasTraits):
    plot = Instance(Component)

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                    resizable=True, title=title,
                    width=size[0], height=size[1]
                    )

    def plotData(self, xdata, ydata):

        # Create the index
        numpoints = 100
        low = -5
        high = 15.0
        x = arange(low, high, (high-low)/numpoints)
        plotdata = ArrayPlotData(x=xdata, y=ydata)

        # Create the left plot
        left_plot = Plot(plotdata)
        left_plot.x_axis.title = "x"
        left_plot.y_axis.title = "y"
        renderer = left_plot.plot(("x", "y"), type="line", color="blue",
                                  width=2.0)[0]
        renderer.overlays.append(LineInspector(renderer, axis='value',
                                               write_metadata=True,
                                               is_listener=True))
        renderer.overlays.append(LineInspector(renderer, axis="index",
                                               write_metadata=True,
                                               is_listener=True))
        left_plot.overlays.append(ZoomTool(left_plot, tool_mode="range"))
        left_plot.tools.append(PanTool(left_plot))

        # Create the right plot
        right_plot = Plot(plotdata)
        right_plot.index_range = left_plot.index_range
        right_plot.orientation = "v"
        right_plot.x_axis.title = "x"
        right_plot.y_axis.title = "y"
        renderer2 = right_plot.plot(("x","y"), type="line", color="red", width=2.0)[0]
        renderer2.index = renderer.index
        renderer2.overlays.append(LineInspector(renderer2, write_metadata=True, is_listener=True))
        renderer2.overlays.append(LineInspector(renderer2, axis="value", is_listener=True))
        right_plot.overlays.append(ZoomTool(right_plot, tool_mode="range"))
        right_plot.tools.append(PanTool(right_plot))

        container = HPlotContainer(background="lightgray")
        container.add(left_plot)
        container.add(right_plot)

        return container


plotwndw = Plotwndw()

if __name__ == "__main__":
    plotwndw.configure_traits()

# EOF
