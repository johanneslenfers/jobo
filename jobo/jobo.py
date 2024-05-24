from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from jobo.bayesian_optimization import BayesianOptimization

from jobo.util import ObjectiveFunction
from jobo.util import SurrogateModel
from jobo.util import AcquisitionFunction


class JoBO: 

    def __init__(self,
                initial_points: int, 
                iterations: int, 
                objective_function: ObjectiveFunction, 
                surrogate_model: SurrogateModel, 
                acquisition_function: AcquisitionFunction,
                plotting_precision: int = 1000
                ):

        # initialize bayesian optimization logic 
        self.bayesian_optimization = BayesianOptimization(
            initial_points = initial_points,
            iterations = iterations,
            objective_function = objective_function,
            surrogate_model = surrogate_model,
            acquisition_function = acquisition_function,
        )

        self.plotting_precision = plotting_precision

    def update_plot(self, version):

        # setup input values for plotting   
        version = int(version)
        x = np.linspace(
            start = self.bayesian_optimization.objective_function.range.min, 
            stop = self.bayesian_optimization.objective_function.range.max, 
            num = self.plotting_precision
            )

        # update evaluated points 
        self.eval_points_plot.set_xdata(
            [elem.x for elem in self.bayesian_optimization.record[version].samples]
        )
        self.eval_points_plot.set_ydata(
            [elem.y for elem in self.bayesian_optimization.record[version].samples]
        )

        # update model data
        self.model_plot.set_ydata(self.bayesian_optimization.record[version].surrogate_model.evaluate(x))

        # update plot 
        self.fig.canvas.draw_idle()

    def slider_plot(self):

        # set up the figure and axis
        self.fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # prepare x axis 
        x = np.linspace(
            start = self.bayesian_optimization.objective_function.range.min, 
            stop = self.bayesian_optimization.objective_function.range.max, 
            num = self.plotting_precision
            )

        # plot objective function, model function and evaluated points  
        y_model = [self.bayesian_optimization.record[0].surrogate_model.evaluate(i) for i in x]
        y_objective = [self.bayesian_optimization.record[0].objective_function.evaluate(i) for i in x]

        self.model_plot, = plt.plot(
            x, 
            y_model, 
            alpha=0.8, 
            label=f'Model'
            )

        self.objective_plot, = plt.plot(
            x, 
            y_objective, 
            alpha=0.8, 
            label='f(x)'
            )

        self.eval_points_plot, = plt.plot(
            [elem.x for elem in self.bayesian_optimization.record[self.bayesian_optimization.initial_points-1].samples],
            [elem.y for elem in self.bayesian_optimization.record[self.bayesian_optimization.initial_points-1].samples],
            'o',
            label = 'observations',
            )

        # set limit of y-axis based on min and max of objective function within given range
        plt.ylim(
            min([self.bayesian_optimization.objective_function.evaluate(i) for i in x]) * 1.1,
            max([self.bayesian_optimization.objective_function.evaluate(i) for i in x]) * 1.1
            )

        # set titles and legend 
        plt.title('Bayesian Optimization')
        plt.get_current_fig_manager().set_window_title('Bayesian Optimization')
        plt.legend()

        # create slider 
        axcolor = 'lightgoldenrodyellow'
        ax.margins(x=0)
        ax_exp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        exp_slider = Slider(ax=ax_exp, label='Iteration', valmin=0, valmax=self.bayesian_optimization.iterations-1, valinit=0, valstep=1.0)
        exp_slider.on_changed(self.update_plot)

        plt.show()

    def run(self):

        # precompute bo 
        self.bayesian_optimization.optimize()

        # plot bo  
        self.slider_plot()