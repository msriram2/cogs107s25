"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

import aesara.tensor as at


#CHATGTP GENERATED TROUBLESHOOTING
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,exception_verbosity=high'


"""
GOAL: Adapt SDT model to quantify the effect of the Stimulus
Type and Trial Difficulty on the participants' performance. 
    - Step 1: Check the convergence of the SDT model (using a delta plot) and 
    display the posterior distributions of the parameter.
    - Step 2: Compare the effects of the Trial Difficulty manipulation with the effects 
    of the Stimulus Type Manipulation. 
"""

"""
Important terms to know: 
- Choice Response Time (CRT): Measurements collected in experiments 
where participants are required to choose between two or more response options 
based on a stimulus --> measures the time taken to make that choice. 
- Diffusion Model: Mathematical framework used to explain how people make decisions between two 
or more choices over time based on noise and uncertain evidence. Accounts for both accuracy 
and reaction time. 
- Delta Plot: A graph that displays response time (RT) distributions based 
on tasks that display conflicting or competing responses. 
- Signal Detection Theory: A framework for how people detect signals
under conditions of uncertainty. 
- Posterior Distribution: The result is a distribution reflecting both 
the prior belief and evidence from the data. Represents an updated belief about 
a parameter or hypothesis after observering data.
"""

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    older_data = pd.read_csv(file_path)

    # Removes NA values 
    old_data = older_data.dropna(how ='any')
    data = old_data.dropna(axis=1, how='any')
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    data['difficulty'] = data['difficulty'].astype(int)
    data['stimulus_type'] = data['stimulus_type'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal', 'difficulty', 'stimulus_type']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'difficulty', 'stimulus_type', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                #CHAT GPT Help: need help defining 'condition' and 'difficulty'
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'difficulty': signal_trials['difficulty'].iloc[0], 
                        'stimulus_type': signal_trials['stimulus_type'].iloc[0],
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)

        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    #CHATGPT GENERATED CODE FOR TROUBLESHOOTING: Trial-level info
    data['pnum'] = data['pnum'].astype("category")
    data['condition'] = data['condition'].astype("category")
    
    p_idx = data['pnum'].cat.codes.values
    c_idx = data['condition'].cat.codes.values
    stimulus_type = data['stimulus_type'].values
    trial_difficulty = data['difficulty'].values 

    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    """ 
    # Get unique participants and conditions
    P = len(data['pnum'].cat.categories)
    C = len(data['condition'].cat.categories)
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level parameters
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)

        # Establish priors for trial difficulty and stimulus type 
        stim_prior = pm.Normal("stim_prior", mu=0.0, sigma=1.0, shape=2)
        diff_prior = pm.Normal('diff_prior', mu=0.0, sigma=1.0, shape=2)

        """
        #Stimulus and trial variables 
        stimulus_type = data['stimulus_type'].values
        trial_difficulty = data['difficulty'].values
        """

        """
        #CHATGPT GENERATED CODE for first line /Help on last two lines: Making stimulus type and trial difficulty PyMC compatible

        comp_condition = pm.Data("comp_condition", condition_idx)
        comp_stim_type = pm.Data("comp_stim_type", stimulus_type)
        comp_trial_diff = pm.Data("comp_trial_diff", trial_difficulty)
        """

        """
        #CHATGPT HELP: Establish mean d' 
        mu_d_prime = mean_d_prime[comp_condition] + stim_prior[comp_stim_type] + diff_prior[comp_trial_diff] 

        print('mu_d_prime shape:', mu_d_prime.eval().shape)
        """

        """
        mu_d_prime = mean_d_prime[c_idx] + stim_prior[stimulus_type] + diff_prior[trial_difficulty] 

        print('mu_d_prime shape:', mu_d_prime.eval().shape)
        """
        #CHATGPT Generated: For Troubleshooting and help with mu_d_prime
        # Initialize a (P, C) matrix of mu_d_prime
        mu_d_prime = at.zeros((P, C))

        # Loop through rows in the data to populate mu_d_prime at [p_idx, c_idx]
        for i in range(len(data)):
            p = p_idx[i]
            c = c_idx[i]
            stim = stim_idx[i]
            diff = diff_idx[i]

            # Expression for this trial
            val = mean_d_prime[c] + stim_prior[stim] + diff_prior[diff]

            # Assign to correct [P, C] cell using `at.set_subtensor`
            mu_d_prime = at.set_subtensor(mu_d_prime[p, c], val)


        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=mu_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        #hit_rate = pm.math.invlogit(d_prime - criterion[data['pnum'] - 1, data['condition']])
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)

        """
        # Likelihood for signal trials
        # Note: pnum is 1-indexed in the data, but needs to be 0-indexed for the model, so we change the indexing here.  The results table will show participant numbers starting from 0, so we need to interpret the results accordingly.
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
        """ 

        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[p_idx, c_idx], 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[p_idx, c_idx], 
                   observed=data['false_alarms'])
                
        
        print("mean_d_prime shape:", C)
        print("Unique comp_condition values:", np.unique(condition_idx))
        print("Max index in comp_condition:", np.max(condition_idx))

        print("Shape of mean_d_prime[comp_condition]:", mean_d_prime[comp_condition].shape)
        print("Shape of stim_prior[comp_stim_type]:", stim_prior[comp_stim_type].shape)
        print("Shape of diff_prior[comp_trial_diff]:", diff_prior[comp_trial_diff].shape)
        
        print("Unique values in comp_condition:", np.unique(comp_condition))
        print("Shape of mean_d_prime:", mean_d_prime.eval().shape)  # only after model context

        #CHATGPT Help
        trace = pm.sample() 

    
    #CHATGPT Help: Checking Convergence using summary stats and a trace plot
    summary = pm.summary(trace, var_names=["mean_d_prime", "stim_type", "trial_diff"])
    print(summary)

    """
    sdt_plot = pm.plot_trace(trace)
    print(sdt_plot)
    posterior = pm.plot_posterior(trace, var_names=["d_prime", "criterion"])
    print(posterior)
    """
    
    return sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    """OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'"""
    OUTPUT_DIR = Path(__file__).parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')

def run_analysis(): 
    file = 'data.csv'
    #part_num = 1
    sdt_new_data = read_data(file, 'sdt') 
    print(sdt_new_data)
    #delta_new_data = read_data(file, 'delta plots')
    new_sdt_model = apply_hierarchical_sdt_model(sdt_new_data)
    #part_num = sdt_new_data["pnum"].value
    #draw_delta_plots(delta_new_data, part_num)
    #print('Successfully Generated Delta Plots')

# Main execution
if __name__ == "__main__":
    """
    file_to_print = Path(__file__).parent / 'README.md'
    with open(file_to_print, 'r') as file:
        print(file.read())
    """
    run_analysis()
    