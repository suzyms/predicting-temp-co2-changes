# main.py
""" 
This module is the main pipeline for predicting temperature changes based on co2
emissions using Tensorflow. It requires the data_loader.py and 
model.py modules.

Functions:
    main_pipeline: preps data, builds and runs the model and plots forcast
    
"""    
    
from data_loader import read_text_file
from model import TempPredict

def main_pipeline():

    file_name = 'co2_mm_mlo.txt'
    skiprows = 53
    column_list = ['year', 'month', 'decimal_date', 'monthly_avg_co2',
            'deseasonalized_co2', 'num_days', 'std_days', 'uncertainty']
    
    modern_co2 = read_text_file(file_name, skiprows=skiprows, column_list=column_list)
    
    file_name = 'Land_and_Ocean_complete.txt'
    skiprows = 2157
    column_list = ['year', 'month', 'temp_anom', 'temp_uncertainty', 'annual_temp_anom',
            'ann_temp_uncertainty', 'five_year_anom', 'five_yr_temp_uncertainty',
            'ten_year_anom', 'ten_yr_temp_uncertainty', 'twenty_year_anom',
            'twenty_yr_temp_uncertainty']
    
    modern_temp = read_text_file(file_name, skiprows=skiprows, column_list=column_list)
    
    
    temperature_prediction = TempPredict(modern_co2, modern_temp)
    
    temperature_prediction.clean_data()
    
    temperature_prediction.prep_data()
    
    temperature_prediction.build_model()
    
    temperature_prediction.train()
    
    temperature_prediction.sample_q()
    
    temperature_prediction.temp_forcast()
    
    temperature_prediction.plot_forcast()
    
    



if __name__ == '__main__':
    main_pipeline()


