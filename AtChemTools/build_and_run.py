#imports
import os
import subprocess
import shutil
import uuid
import pandas as pd
import numpy as np
from .species_from_mechanism import return_all_species
from .utilities import is_number
import warnings
from datetime import datetime

warnings.simplefilter('always', UserWarning)

def wipe_file(file_path : str):
    """Clears the contents of the specified file"""
    open(file_path, 'w').close()

def list_to_config_file(in_list : list, filepath : str):
    """Converts a list to an AtChem2 configuration file with each list item 
    written on a new line."""

    lines = [f"{x}\n" for x in in_list]
    with open(filepath,"w") as file:
        file.writelines(lines)

def series_to_config_file(in_series : pd.Series, filepath : str, 
                          round_nums : bool = True):
    """Converts a pandas series to an AtChem2 configuration file with indices and
    values written on each line.
    
    If `round_nums` is true, then the series values will be formatted as `.5e` 
    to prevent long numbers exceeding the FORTRAN line length limit in AtChem2."""
    if round_nums:
        lines = [f"{x} {y}\n" for x,y in in_series.items()]
    else:
        lines = [f"{x} {y:.5e}\n" for x,y in in_series.items()]
        
    with open(filepath,"w") as file:
        file.writelines(lines)
        
def dataframe_to_config_files(in_df : pd.DataFrame, dirpath : str, 
                              round_nums : bool = True):
    """Converts each column of a pandas dataframe to an AtChem2 configuration 
    file with indices and values written on each line. The name of each 
    configuration file produced will match the name of the column.
    
    If `round_nums` is true, then the column values will be formatted as `.5e` 
    to prevent long numbers exceeding the FORTRAN line length limit in AtChem2."""
    for col in in_df.columns:
        if round_nums:
            lines = [f"{x} {y}\n" for x,y in in_df[col].dropna().items()]
        else:
            lines = [f"{x} {y:.5e}\n" for x,y in in_df[col].dropna().items()]
            
        with open(dirpath+os.sep+col,"w") as file:
            file.writelines(lines)
        
def write_config(model_path : str, initial_concs : pd.Series = pd.Series(), 
                 spec_constrain : pd.DataFrame = pd.DataFrame(), 
                 spec_constant : pd.Series = pd.Series(), 
                 env_constrain : pd.DataFrame = pd.DataFrame(), 
                 photo_constant : pd.Series = pd.Series(),
                 photo_constrain : pd.DataFrame = pd.DataFrame(),
                 env_vals : pd.Series = pd.Series(["298.15", "1013.25", "NOTUSED",
                                                   "3.91e+17", "0.41", "NOTUSED", 
                                                   "NOTUSED", "NOTUSED", "OPEN", 
                                                   "NOTUSED"], 
                                                  index = ["TEMP", "PRESS", 
                                                           "RH", "H2O", "DEC", 
                                                           "BLHEIGHT", "DILUTE", 
                                                           "JFAC", "ROOF", "ASA"]),
                 spec_output : list = [], rate_output : list = []):
    """Prepares model files in a specified AtChem2 directory for building and 
    running, filling them out with the provided input data"""
    
    #initialConcentrations.config
    series_to_config_file(initial_concs.dropna(), 
                        f"{model_path}/configuration/initialConcentrations.config")
        
    #speciesConstrained.config
    list_to_config_file(list(spec_constrain.columns), 
                        f"{model_path}/configuration/speciesConstrained.config")
    #species constraints
    dataframe_to_config_files(spec_constrain,f"{model_path}/constraints/species/")
        
    #photolysisConstrained.config
    list_to_config_file(list(photo_constrain.columns), 
                        f"{model_path}/configuration/photolysisConstrained.config")
    #photolysis constraints
    dataframe_to_config_files(photo_constrain,f"{model_path}/constraints/photolysis/")

    #speciesConstant.config
    series_to_config_file(spec_constant.dropna(), 
                          f"{model_path}/configuration/speciesConstant.config")
    
    #photolysisConstant.config 
    #make lines for config file which has to be in the format "1 1E-4 J1" etc.
    pconst_lines = [f"{k.strip('J')} {v} {k}" for k,v in photo_constant.items()]
    
    list_to_config_file(pconst_lines,
                        f"{model_path}/configuration/photolysisConstant.config")
    
    
    #environmentVariables.config
    env_copy = env_vals.copy()
    default_env = pd.Series(["298.15", "1013.25", "NOTUSED", "3.91e+17", "0.41", 
                             "NOTUSED", "NOTUSED", "NOTUSED", "OPEN", "NOTUSED"], 
                            index = ["TEMP", "PRESS", "RH", "H2O", "DEC", 
                                     "BLHEIGHT", "DILUTE", "JFAC", "ROOF", "ASA"])
    
    #Ensure that H2O is set to 'CALC' if RH is specified, and that RH is set to
    #'NOTUSED' if H2O is specified then. Raise an error if both are specified.
    if ("RH" in env_copy.index) and ("H2O" in env_copy.index):
        if ((is_number(env_copy["H2O"]) and (env_copy["RH"] != "NOTUSED")) or
            ((is_number(env_copy["RH"]) and (env_copy["H2O"] != "CALC")))):
                raise Exception("Both RH and H2O specified for environmentVariables.config. Specify only one, or set the unused variable to 'NOTUSED' or 'CALC' for RH and H2O, respectively.")
    elif ("RH" in env_copy.index):
        env_copy["H2O"] = "CALC"
    elif ("H2O" in env_copy.index):
        env_copy["RH"] = "NOTUSED"
    
    #fill in any missing environment variable values with defaults (if they 
    # aren't supposed to be constrained)
    for k,v in default_env.items():
        if k not in env_copy.index:
            if k in env_constrain.index:
                env_copy[k] = "CONSTRAINED"
            else:
                env_copy[k] = v
        
    env_var_lines=f"""1 TEMP			{env_copy['TEMP']}
    2 PRESS			{env_copy['PRESS']}
    3 RH			{env_copy['RH']}
    4 H2O			{env_copy['H2O']}
    5 DEC			{env_copy['DEC']}
    6 BLHEIGHT		{env_copy['BLHEIGHT']}
    7 DILUTE		{env_copy['DILUTE']}
    8 JFAC			{env_copy['JFAC']}
    9 ROOF			{env_copy['ROOF']}
    10 ASA          {env_copy['ASA']}"""  
    
    #append any custom environment variables
    for i,k in enumerate([x for x in env_copy.index if x not in default_env.index]):
        env_var_lines += f"\n{11+i} {k} {env_copy[k]}"
    
    with open(f"{model_path}/configuration/environmentVariables.config","w") as file:
        file.write(env_var_lines)

    #environment constraints
    for col in env_constrain.columns:
        if env_copy[col] == "CONSTRAINED":
            if col != "JFAC":
                env_path = f"{model_path}/constraints/environment/{col}"
            else:
                env_path = f"{model_path}/constraints/photolysis/{col}"
            series_to_config_file(env_constrain[col].dropna(), env_path)
        else:
            raise Exception(f"Constraint provided for {col}, but value is set as {env_copy[col]} not 'CONSTRAINED'")   
        
    #outputSpecies.config and outputRates.config
    list_to_config_file(spec_output,
                        f"{model_path}/configuration/outputSpecies.config")
    list_to_config_file(rate_output,
                        f"{model_path}/configuration/outputRates.config")    
    
def write_model_params(model_path : str, nsteps : int, model_tstep : int, 
                       tstart : int, day : int, month : int, year : int, 
                       lat : float, lon : float):
    """Write the provided input data to the model parameters file of the 
    specified AtChem2 directory."""
    
    model_params_lines=f"""{nsteps}			number of steps
    {model_tstep}			step size (seconds)
    2			species interpolation method (pw constant = 1, pw linear = 2)
    2			conditions interpolation method (pw constant = 1, pw linear = 2)
    {model_tstep}			rates output step size (seconds)
    {tstart}			model start time (seconds)
    0			jacobian output step size (seconds)
    {lat}			latitude (degrees)
    {lon}			longitude (degrees)
    {day:02d}			day
    {month:02d}			month
    {year:04d}			year
    {model_tstep}			reaction rates output step size (seconds)"""

    with open(model_path+"/configuration/model.parameters","w") as file:
        file.write(model_params_lines)
    

def build_model(atchem2_path : str, mechanism_path : str, model_path : str = ""):
    """Builds the specified AtChem2 model, ready for running"""
    script_dir = os.getcwd()
    os.chdir(atchem2_path)
    subprocess.run(
        [f"{atchem2_path}/build/build_atchem2.sh", mechanism_path,
         f"{model_path}/configuration/"],
        check=True,
    )
    os.chdir(script_dir)

def run_model(atchem2_path : str, model_path : str = ""):
    """Runs the specified (pre-built) AtChem2 model"""
    script_dir = os.getcwd()
    os.chdir(atchem2_path)
    if model_path:
        subprocess.run([f"{atchem2_path}/atchem2", f"--model={model_path}"], check=True)
    else:
        subprocess.run([f"{atchem2_path}/atchem2"], check=True)
    os.chdir(script_dir)

def find_unique_dirname(atchem2_path : str):
    """Creates a unique model sub-directory name based on the current datetime, 
    process ID, and a short random component. This should avoid over-writing 
    existing model sub-directories when running a new simulation, and give a 
    meaningful model subdirectory name for the user, if needed."""
    
    fmt_dtime = str(datetime.now()).replace(" ", "_").replace(":", "-")
    unique_id = uuid.uuid4().hex[:8]

    return f"model_{fmt_dtime}_{os.getpid()}_{unique_id}"
                        
def _write_build_run_injections(injection_df : pd.DataFrame, atchem2_path : str, 
                                mech_path : str, day : int, 
                                month : int, year : int, t_start : int, 
                                t_end : int, step_size : int, 
                                initial_concs : pd.Series, spec_constrain : pd.DataFrame, 
                                spec_constant : pd.Series, env_constrain : pd.DataFrame, 
                                photo_constant : pd.Series, photo_constrain : pd.DataFrame, 
                                env_vals : pd.Series, spec_output : list, 
                                rate_output : list, lat : float, lon : float,
                                keep_rundirs : bool):
    """Called by the 'write_build_run' function to configure, build and run
    a specified AtChem2 model including instantaneous increases in 
    concentrations of certain species. """
    
    #dataframe to store model outputs
    stitched_output = pd.DataFrame(dtype=float)
    #dataframes to store rate outputs
    stitched_loss_rates = pd.DataFrame(dtype=float)
    stitched_prod_rates = pd.DataFrame(dtype=float)
    #dataframe to store environment variables
    stitched_env = pd.DataFrame(dtype=float)
    #dataframe to store photolysis rates
    stitched_photo = pd.DataFrame(dtype=float)
    
   
    #make a list of ordered injection times to iterate through
    ordered_times = injection_df.index.to_list()
    ordered_times.sort()
    
    #remove any injections outside of the start and end times and add the start
    #time to the injection list
    ordered_times = [x for x in ordered_times if (x > t_start) and (x < t_end)]
    ordered_times.insert(0, t_start)
    
    all_specs = return_all_species(mech_path)
    
    for i,inj_time in enumerate(ordered_times):
        #copy atchem2 model directory 
        new_model_dir = find_unique_dirname(atchem2_path)
        new_model_path = f"{atchem2_path}/{new_model_dir}"
        shutil.copytree(f"{atchem2_path}/model", new_model_path)
        #copy the mechanism to the AtChem directory
        new_mech_path = f"{new_model_path}/{mech_path.split('/')[-1]}"
        shutil.copy(mech_path, new_mech_path)
        
        #write config files using data passed
        write_config(new_model_path, initial_concs=initial_concs, 
                     spec_constrain=spec_constrain, spec_constant=spec_constant,
                     env_constrain=env_constrain, env_vals=env_vals, 
                     photo_constant = photo_constant, photo_constrain = photo_constrain,
                     spec_output=all_specs,
                     rate_output=all_specs) #return all species for now (all needed to set the new start concs)

        
        if (i != (len(ordered_times)-1)): #if this isn't the last iteration then 
        #calculate the next injection time, otherwise the next injection time
        #is just the model end time
            next_injtime = ordered_times[i+1]
        else:
            next_injtime = t_end
            
        if i != 0: #if it isn't the first run, then adjust concentrations based on required injections
        #rewrite initial concentrations file to match the model output from
        #the previous model run
            new_start_concs = stitched_output.loc[min(stitched_output.index, key=lambda x:abs(x-inj_time))] #species concentrations at the closest time to the previous injection time
            
            #change the start concentrations for species injected this time
            specs = injection_df.loc[inj_time].dropna().index.to_list()
            for s in specs:
                if s != "NOx":
                    new_start_concs.loc[s] = injection_df.loc[inj_time, s]
                else:
                    #for the NOx constraint, calculate the NO/NO2 ratio 
                    #and change NOx such that the ratio is preserved.
                    old_no = new_start_concs.loc["NO"]
                    old_no2 = new_start_concs.loc["NO2"]
                    
                    old_total_nox = old_no + old_no2
                    nox_deficit = injection_df.loc[inj_time, s] - old_total_nox                             
                    
                    new_no = (old_no + (nox_deficit*(old_no/old_total_nox)))
                    new_no2 = (old_no2 + (nox_deficit*(old_no2/old_total_nox)))
                    
                    new_start_concs.loc["NO"] = new_no
                    new_start_concs.loc["NO2"] = new_no2
            
            init_lines = [f"{k} {v}\n" for k,v in new_start_concs.to_dict().items()]
            
            with open(new_model_path+"/configuration/initialConcentrations.config",
                      "w") as file:
                file.writelines(init_lines)
        
    
        #rewrite model parameters file to only run for the length of the 
        #injection of interest
        model_length=(next_injtime+step_size) - inj_time
        nsteps=int(model_length/step_size)
               
        write_model_params(new_model_path, nsteps, step_size, inj_time, day, 
                           month, year, lat=lat, lon=lon)
        
        #build and run the model
        build_model(atchem2_path, new_mech_path, new_model_dir)
        run_model(atchem2_path, new_model_dir)
        
        #read the model output and append it to the stitched df
        output = pd.read_csv(f"{new_model_path}/output/speciesConcentrations.output", 
                             index_col=0, sep='\s+')
        loss_output = pd.read_csv(f"{new_model_path}/output/lossRates.output", 
                                  sep='\s+',
                                  keep_default_na=False)
        prod_output = pd.read_csv(f"{new_model_path}/output/productionRates.output", 
                                  sep='\s+',
                                  keep_default_na=False)
        env_output = pd.read_csv(f"{new_model_path}/output/environmentVariables.output", 
                                 index_col=0, sep='\s+')
        photo_output = pd.read_csv(f"{new_model_path}/output/photolysisRates.output", 
                                 index_col=0, sep='\s+')

        #trim off the values that are accounted for by subsequent iterations
        output = output.iloc[:-1,:]
        loss_output = loss_output.loc[loss_output["time"]!=(next_injtime+step_size),:]
        prod_output = prod_output.loc[prod_output["time"]!=(next_injtime+step_size),:]
        env_output = env_output.iloc[:-1,:]
        photo_output = photo_output.iloc[:-1,:]
                
        stitched_output = pd.concat([stitched_output, output])
        stitched_loss_rates = pd.concat([stitched_loss_rates, loss_output])
        stitched_prod_rates = pd.concat([stitched_prod_rates, prod_output])
        stitched_env = pd.concat([stitched_env, env_output])
        stitched_photo = pd.concat([stitched_photo, photo_output])
    
        #remove model directory (unless requested to keep)
        if not keep_rundirs:
            shutil.rmtree(new_model_path)
        
    #select only the output speices
    stitched_output = stitched_output[spec_output]
    stitched_loss_rates = stitched_loss_rates[stitched_loss_rates["speciesName"].isin(rate_output)]
    stitched_prod_rates = stitched_prod_rates[stitched_prod_rates["speciesName"].isin(rate_output)]

    return (stitched_output,stitched_loss_rates,stitched_prod_rates,stitched_env, stitched_photo)
    
    
def _write_build_run_nox_constraint(nox_series : pd.Series, atchem2_path : str, 
                                    mech_path : str, day : int, 
                                    month : int, year : int, t_start : int, 
                                    t_end : int, step_size : int, 
                                    initial_concs : pd.Series, spec_constrain : pd.DataFrame, 
                                    spec_constant : pd.Series, env_constrain : pd.DataFrame, 
                                    photo_constant : pd.Series, photo_constrain : pd.DataFrame, 
                                    env_vals : pd.Series, spec_output : list, 
                                    rate_output : list, lat : float, lon : float,
                                    keep_rundirs : bool):
    """Called by the 'write_build_run' function to configures, build and run
    a specified AtChem2 model including a constraint on total NOx, while NO 
    and NO2 are allowed to vary freely.
    """
    warnings.warn("""WARNING. THIS NOX CONSTRAINT FEATURE IS EXPERIMENTAL,
CHECK ANY MODEL OUTPUT THOROUGHLY TO ENSURE THE RESULTS ARE AS EXPECTED.
THE NOX CONSTRAINT FEATURE IS ALSO VERY SLOW AS IT REQUIRES THE REPEATED
BUILDING OF MANY INDIVIDUAL MODELS.""")
    
    #dataframe to store model outputs
    stitched_output = pd.DataFrame(dtype=float)
    #dataframes to store rate outputs
    stitched_loss_rates = pd.DataFrame(dtype=float)
    stitched_prod_rates = pd.DataFrame(dtype=float)
    #dataframe to store environment variables
    stitched_env = pd.DataFrame(dtype=float)
    #dataframe to store photolysis rates
    stitched_photo = pd.DataFrame(dtype=float)
   
    #calculate the number of timesteps the model must run for
    model_length = t_end - t_start
    nsteps=int(model_length/step_size)
    
    all_specs = return_all_species(mech_path)
    
    #interpolate the nox series to ensure there are values for every model time
    nox_series_interp = nox_series.loc[t_start:t_end]
    nox_series_interp = nox_series_interp.reindex(np.arange(t_start, 
                                                            t_end+step_size, 
                                                            step_size))
    nox_series_interp = nox_series_interp.interpolate()
    
    for istep in range(nsteps):
        step_time = t_start + (istep*step_size)
        
        #copy atchem2 model directory 
        new_model_dir = find_unique_dirname(atchem2_path)
        new_model_path = f"{atchem2_path}/{new_model_dir}"
        shutil.copytree(f"{atchem2_path}/model", new_model_path)

        #copy the mechanism to the AtChem directory
        new_mech_path = f"{new_model_path}/{mech_path.split('/')[-1]}"
        shutil.copy(mech_path, new_mech_path)
        
        #write config files using data passed
        write_config(new_model_path, initial_concs=initial_concs, 
                     spec_constrain=spec_constrain, spec_constant=spec_constant,
                     env_constrain=env_constrain, env_vals=env_vals, 
                     photo_constant = photo_constant, photo_constrain = photo_constrain,
                     spec_output=all_specs,
                     rate_output=all_specs) #return all species for now (all needed to set the new start concs)

        
        if istep != 0: #if it isn't the first run, then adjust concentrations based on required injections
        #rewrite initial concentrations file to match the model output from
        #the previous model run
            new_start_concs = stitched_output.iloc[-1] #species concentrations at the last time step
            
            #for the NOx constraint, calculate the NO/NO2 ratio 
            #and change NOx such that the ratio is preserved.
            old_no = new_start_concs.loc["NO"]
            old_no2 = new_start_concs.loc["NO2"]
            
            old_total_nox = old_no + old_no2
            nox_deficit = nox_series_interp[step_time] - old_total_nox                             
            
            new_no = (old_no + (nox_deficit*(old_no/old_total_nox)))
            new_no2 = (old_no2 + (nox_deficit*(old_no2/old_total_nox)))
            
            new_start_concs.loc["NO"] = new_no
            new_start_concs.loc["NO2"] = new_no2
            
            init_lines = [f"{k} {v}\n" for k,v in new_start_concs.to_dict().items()]
            
            with open(new_model_path+"/configuration/initialConcentrations.config",
                      "w") as file:
                file.writelines(init_lines)
        else: #just check that we have some NOx in the model if this is the first step
            if not any([x in initial_concs.keys() for x in ["NO","NO2"]]):
                #if there is not initial NO or NO2 specified, then split the
                #given NOx value 50:50 between NO and NO2
                half_val = nox_series_interp[step_time]/2
                
                with open(new_model_path+"/configuration/initialConcentrations.config",
                          "a") as file:
                    file.write(f"NO2 {half_val}\nNO {half_val}")
            
    
        #rewrite model parameters file to only run for the length of the 
        #injection of interest
               
        write_model_params(new_model_path, 1, step_size, step_time, day, 
                           month, year, lat=lat, lon=lon)
        
        #build and run the model
        build_model(atchem2_path, new_mech_path, new_model_dir)
        run_model(atchem2_path, new_model_dir)
        
        #read the model output and append it to the stitched df
        output = pd.read_csv(f"{new_model_path}/output/speciesConcentrations.output", 
                             index_col=0, sep='\s+')
        loss_output = pd.read_csv(f"{new_model_path}/output/lossRates.output", 
                                  sep='\s+',
                                  keep_default_na=False)
        prod_output = pd.read_csv(f"{new_model_path}/output/productionRates.output", 
                                  sep='\s+',
                                  keep_default_na=False)
        env_output = pd.read_csv(f"{new_model_path}/output/environmentVariables.output", 
                                 index_col=0, sep='\s+')
        photo_output = pd.read_csv(f"{new_model_path}/output/photolysisRates.output", 
                                 index_col=0, sep='\s+')

        #trim off the first value (if this isn't the first step)
        if istep != 0:
            output = output.iloc[1:,:]
            env_output = env_output.iloc[1:,:]
            photo_output = photo_output.iloc[1:-1,:]

        stitched_output = pd.concat([stitched_output, output])
        stitched_loss_rates = pd.concat([stitched_loss_rates, loss_output])
        stitched_prod_rates = pd.concat([stitched_prod_rates, prod_output])
        stitched_env = pd.concat([stitched_env, env_output])
        stitched_photo = pd.concat([stitched_photo, photo_output])

        #remove model directory (unless requested to keep)
        if not keep_rundirs:
            shutil.rmtree(new_model_path)
        
    #select only the output speices
    stitched_output = stitched_output[spec_output]
    stitched_loss_rates = stitched_loss_rates[stitched_loss_rates["speciesName"].isin(rate_output)]
    stitched_prod_rates = stitched_prod_rates[stitched_prod_rates["speciesName"].isin(rate_output)]
    
    return (stitched_output,stitched_loss_rates,stitched_prod_rates,stitched_env, stitched_photo)
    
def write_build_run(atchem2_path : str, mech_path : str, day : int, month : int, 
                    year : int, t_start : int, t_end : int, lat : float, 
                    lon : float, step_size : int, 
                    initial_concs : pd.Series = pd.Series(), 
                    spec_constrain : pd.DataFrame = pd.DataFrame(), 
                    spec_constant : pd.Series = pd.Series(), 
                    env_constrain : pd.DataFrame = pd.DataFrame(), 
                    photo_constant : pd.Series = pd.Series(), 
                    photo_constrain : pd.DataFrame = pd.DataFrame(), 
                    env_vals : pd.Series = pd.Series(["298.15", "1013.25", "NOTUSED",
                                                      "3.91e+17", "0.41", "NOTUSED", 
                                                      "NOTUSED", "NOTUSED", "OPEN", 
                                                      "NOTUSED"], 
                                                     index = ["TEMP", "PRESS", 
                                                              "RH", "H2O", "DEC", 
                                                              "BLHEIGHT", "DILUTE", 
                                                              "JFAC", "ROOF", "ASA"]),
                    spec_output : list = [], rate_output : list = [], keep_rundirs : bool = False,
                    injection_df : pd.DataFrame = pd.DataFrame, nox_series : pd.Series = pd.Series):
    """Configures, builds and runs a specified AtChem2 model. 
    
    If `injection_df` is specified, then a series of models 
    will be run to simulate a chamber experiments with the (instantaneous) 
    introduction of species into the chamber mid-experiment. 
    injection_df should be a pandas dataframe with columns for each species 
    which undergoes injections and a time index in model time. Each species 
    should then have concentration values defined at the required injection 
    times, with all other time values being NaNs.


    If nox_series is specified, then a series of one step models 
    will be run with NO and NO2 concentrations adjusted after each, to produce 
    a continuous model output with concentrations of NOx constrained (while NO
    and NO2 can vary freely). `nox_series` should be a pandas series with a 
    model-time index, and defined NOx conentrations for each time.
    The NOx concentrations will be linearly interpolated along all of the model
    timesteps.
    WARNING. THIS NOX CONSTRAINT FEATURE IS EXPERIMENTAL AND ALSO VERY SLOW. 
    CHECK ANY MODEL OUTPUT THOROUGHLY TO ENSURE THE RESULTS ARE AS EXPECTED.
    """
    
    if (not injection_df.empty) and (not nox_series.empty):
        raise Exception("""Cannot run models using both species injections and 
                        NOx constraints. Select either injection_dict or 
                        nox_dict arguments, not both.""")
    elif not injection_df.empty:
        return _write_build_run_injections(injection_df = injection_df, 
                                           atchem2_path = atchem2_path, 
                                           mech_path = mech_path, 
                                           day = day, 
                                           month = month, 
                                           year = year, 
                                           t_start = t_start, 
                                           t_end = t_end, 
                                           step_size = step_size,
                                           initial_concs = initial_concs,
                                           spec_constrain = spec_constrain,
                                           spec_constant = spec_constant,
                                           env_constrain = env_constrain,
                                           photo_constant = photo_constant, 
                                           photo_constrain = photo_constrain,
                                           env_vals = env_vals,
                                           spec_output = spec_output,
                                           rate_output = rate_output,
                                           lat = lat,
                                           lon = lon, keep_rundirs = keep_rundirs)
    elif not nox_series.empty:
        return _write_build_run_nox_constraint(nox_series = nox_series, 
                                               atchem2_path = atchem2_path, 
                                               mech_path = mech_path, 
                                               day = day, 
                                               month = month, 
                                               year = year, 
                                               t_start = t_start, 
                                               t_end = t_end, 
                                               step_size = step_size,
                                               initial_concs = initial_concs,
                                               spec_constrain = spec_constrain,
                                               spec_constant = spec_constant,
                                               env_constrain = env_constrain,
                                               photo_constant = photo_constant, 
                                               photo_constrain = photo_constrain,
                                               env_vals = env_vals,
                                               spec_output = spec_output,
                                               rate_output = rate_output,
                                               lat = lat,
                                               lon = lon, keep_rundirs = keep_rundirs)
    else:
    
        #copy atchem2 model directory 
        new_model_dir = find_unique_dirname(atchem2_path)
        new_model_path = f"{atchem2_path}/{new_model_dir}"
        shutil.copytree(f"{atchem2_path}/model", new_model_path)

        #copy the mechanism to the AtChem directory
        new_mech_path = f"{new_model_path}/{mech_path.split('/')[-1]}"
        shutil.copy(mech_path, new_mech_path)

        #write config files using data passed
        write_config(new_model_path, initial_concs=initial_concs, 
                     spec_constrain=spec_constrain, spec_constant=spec_constant,
                     env_constrain=env_constrain, env_vals=env_vals, 
                     photo_constant = photo_constant, photo_constrain = photo_constrain,
                     spec_output=spec_output,
                     rate_output=rate_output)
        
        #change model parameters
        model_length=t_end-t_start
        nsteps=int(model_length/step_size)
    
        write_model_params(new_model_path, nsteps, step_size, t_start, day, 
                           month, year, lat=lat, lon=lon)
        
        #build and run the model
        build_model(atchem2_path, new_mech_path, new_model_dir)
        run_model(atchem2_path, new_model_dir)
        
        #read the model output 
        output = pd.read_csv(f"{new_model_path}/output/speciesConcentrations.output", 
                             index_col=0, sep='\s+')
        
        loss_output = pd.read_csv(f"{new_model_path}/output/lossRates.output", 
                                  sep='\s+',
                                  keep_default_na=False)
        prod_output = pd.read_csv(f"{new_model_path}/output/productionRates.output", 
                                  sep='\s+',
                                  keep_default_na=False)
        
        env_output = pd.read_csv(f"{new_model_path}/output/environmentVariables.output", 
                                 index_col=0, sep='\s+')
        
        photo_output = pd.read_csv(f"{new_model_path}/output/photolysisRates.output", 
                                 index_col=0, sep='\s+')
        
        #remove model directory (unless requested to keep)
        if not keep_rundirs:
            shutil.rmtree(new_model_path)
        
        return (output, loss_output, prod_output, env_output, photo_output)

