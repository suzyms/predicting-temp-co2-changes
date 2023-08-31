# model.py
"""
This module contains the model used to predict monthly 
temperature anomalies from co2 emissions.

Classes: 
    TempPredict: Class to build and visualise a linear temperature 
    prediction model based on co2 emissions using Tensorflow.
    
"""    
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TempPredict:
    """
    Class to build and visualise a linear temperature 
    prediction model based on co2 emissions using Tensorflow.
    
    Attributes:
        self.modern_co2: monthly co2 levels
        self.modern_temp: monthly temperatures 
       
    Methods:
        clean_data: align relevant data columns in time and calculate anomalies
        prep_data: create test/ train split
        build_model: build model parameters 
        train: train the model and generate variational loss curve
        sample_q: calculate mean and sd of parameters sampled from the variational posteriors
        temp_forcast: construct multiple forcasts and take the mean and sd
        plot_forcast: plot true temperature vs forcasted temperature
        
    """
    def __init__(self, modern_co2, modern_temp):
        
        self.modern_co2 = modern_co2
        self.modern_temp = modern_temp
  
        
        
    def clean_data(self):
        """ Align relevant data columns in time and calculate anomalies"""
        
        co2_present = self.modern_co2['deseasonalized_co2']
        monthly_temperature = self.modern_temp['temp_anom']
        temperature_time = self.modern_temp['year'] + (self.modern_temp['month'] - 1) / 12

        start_time = np.where(temperature_time == self.modern_co2['year'][0] + 
                              (self.modern_co2['month'][0] - 1) / 12)[0][0]

        self.monthly_temperature = monthly_temperature[start_time:]
        self.temperature_time = temperature_time[start_time:]

        self.co2_present = co2_present[:len(monthly_temperature)]

        self.carbon_anomaly = co2_present - np.mean(co2_present)
        
    def prep_data(self):
        """ Creat train/test split """
        self.test_set_data_points = 12*10
        self.monthly_temperature_training_set = self.monthly_temperature[:-self.test_set_data_points]
        self.monthly_temperature_test_set = self.monthly_temperature[-self.test_set_data_points:]
        
    def build_model(self):
        """ Build model parameters """
        self.carbon_effect = tfp.sts.LinearRegression(
        design_matrix=tf.reshape(self.carbon_anomaly, (-1, 1)), name='carbon_effect')

        self.model = tfp.sts.Sum([self.carbon_effect,],
                    observed_time_series=self.monthly_temperature_training_set)

        self.variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
                                    model=self.model)

        self.num_variational_steps = int(200)
        self.optimizer = tf.optimizers.Adam(learning_rate=.1)
        
        

    @tf.function(experimental_compile=True)
    def train(self):
        """ Train the model and generate the variational loss curve """
        loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self.model.joint_distribution(
            observed_time_series=self.monthly_temperature_training_set).log_prob,
            surrogate_posterior=self.variational_posteriors,
            optimizer=self.optimizer,
            num_steps=self.num_variational_steps)
        
        return loss_curve
    
        
    
    def sample_q(self):
        """ Calculate mean and sd of parameters sampled from the variational posteriors """
        self.q_samples_ = self.variational_posteriors.sample(50)
        print("Inferred parameters:")
        for param in self.model.parameters:
            print("{}: {} +- {}".format(param.name,
                              np.mean(self.q_samples_[param.name], axis=0),
                              np.std(self.q_samples_[param.name], axis=0)))
            
    
    
    def temp_forcast(self):
        """ Construct multiple forcasts and take the mean and sd """
        temperature_forecast_dist = tfp.sts.forecast(
            model=self.model,
            observed_time_series=self.monthly_temperature_training_set,
            parameter_samples=self.q_samples_,
            num_steps_forecast=self.test_set_data_points)

        num_samples=20

        (
            self.temperature_forecast_mean,
            self.temperature_forecast_scale,
            self.temperature_forecast_samples
        ) = (
            temperature_forecast_dist.mean().numpy()[..., 0],
            temperature_forecast_dist.stddev().numpy()[..., 0],
            temperature_forecast_dist.sample(num_samples).numpy()[..., 0]
            )         


    def plot_forcast(self):
        """ Plot true temperature vs forcasted temperature (mean and sd) """
        colors = sns.color_palette()
        c1, c2 = colors[0], colors[1]
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        num_steps = len(self.temperature_time)
        num_steps_forecast = self.temperature_forecast_mean.shape[-1]
        num_steps_train = num_steps - num_steps_forecast


        ax.plot(self.temperature_time, self.monthly_temperature, lw=2, color=c1, 
                label='ground truth')

        ax.plot(self.temperature_time[-num_steps_forecast:], self.temperature_forecast_samples.T, 
                lw=1, color=c2, alpha=0.1)

        ax.plot(self.temperature_time[-num_steps_forecast:], self.temperature_forecast_mean, 
                lw=2, ls='--', color=c2, label='forecast')

        ax.fill_between(self.temperature_time[-num_steps_forecast:],
                        self.temperature_forecast_mean-2*self.temperature_forecast_scale,
                        self.temperature_forecast_mean+2*self.temperature_forecast_scale, 
                        color=c2, alpha=0.2)

        ymin = min(np.min(self.temperature_forecast_samples), np.min(self.monthly_temperature))
        ymax = max(np.max(self.temperature_forecast_samples), np.max(self.monthly_temperature))
                 
        yrange = ymax-ymin
        ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
        ax.set_title("Global Average Temperature Anomaly")
        ax.legend()
        plt.show()
       
        
        
        
        
