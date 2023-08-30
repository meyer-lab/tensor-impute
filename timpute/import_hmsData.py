import numpy as np
import pandas as pd
import os


def separate_cellLines():
    """Given the path to the object level data fron the dte drop paper, it separates data for each cell line to a dictionary, with cell line names as keys and dataframe of the data for that cell line as the value of the dictionary."""

    cwd = f'{os.getcwd()}/timpute/data'
    df = pd.read_csv(f"{cwd}/fractions.csv", index_col=0)

    df["agent_conc"] = df["target"].astype(str) + "_" + df["agent"].astype(str) + "_" + np.round(np.log10(df["concentration"]), 5).astype(str)

    cellLine_list = list(df['cell_line'].unique())
    df_dict = {}
    for i, val in enumerate(cellLine_list):
        tmp_df = df.loc[df['cell_line'] == val]
        df_dict[val] = tmp_df

    return df_dict

def import_cellLine(cellline_name: str, cellline_df):
    """Choose between the following list to import the data for that specific cell line, then pass the dataframe to this function to get the tensor of cell numbers.
    ['AU565', 'PDX1328', 'BT20', 'BT549', 'HCC1806', 'HCC1954', 'HCC70', 'MCF10A', 'MDAMB231', 'SUM1315', 'SUM149', 'SUM159', 'HTERTHME1', 'SUM190PT', 'SUM44PE', 'MDAMB330', 'SUM185PE', 'SUM229PE', 'SUM52PE', 'UACC893', 'ZR751', 'MCF12A', 'EFM19', 'ZR7530', 'BT474', 'HS578T', 'MCF7', 'T47D', 'MGH312', 'CAL120', 'CAL51', 'CAL851', 'CAMA1', 'MCF10A (GM)', 'SKBR3', 'MDAMB361', 'MDAMB436', 'MDAMB453', 'MDAMB468', 'HCC1143', 'HCC1395', 'HCC1419', 'HCC1937', 'HCC38', '184A1', 'HCC202', 'MDAMB175VII', 'MDAMB415', 'UACC812', 'EVSAT', 'HCC1187', 'HCC1569', 'PDX1206', 'PDX1258', 'PDXHCI002']
    Returns
    -------
    tensor : np.ndarray
        The tensor output is a 4 x 2 x 4 x 12 x 58 which is 4 phases [G1, S, G2M, D], for 2 time points [2, 72], 
        4 replicates, though in some cases we have 3 or 2 replicates, 12 drug concentrations, 
        where the first one is the control as zero, and 58 agents.
    agents : list
        The list of agents with the same order as the last dimension of the tensor.
    concentrations : list[list]
        The range of concentrations for each agent.
    """

    # import the baseline file for control
    cwd = f'{os.getcwd()}/timpute/data'
    ctr = pd.read_csv(f"{cwd}/baseline_fractions.csv")

    # agents = list(cellline_df['agent'].unique())
    agents = ['AZD1775', 'AZD2014', 'AZD5363', 'AZD6738', 'BJP-6-5-3', 'BMS-265246', 'BSJ-01-175', 'BSJ-03-123', 'BSJ-03-124', 'BVD523',
              'FMF-03-146-1', 'FMF-04-107-2', 'FMF-04-112-1', 'Flavopiridol', 'GSK2334470', 'LEE011/Ribociclib', 'LY3023414', 'Pin1-3',
              'R0-3306',  'Rucaparib', 'SHP099', 'THZ-P1-2', 'THZ-P1-2R', 'THZ1', 'THZ531', 'YKL-5-124', 'ZZ1-33B', 'senexin b']
    
    concentrations, targets = [], []
    tensor = np.zeros((4, 2, 4, 10, len(agents))) # first dim = G1, S, G2M, abnormal/dead 
    for i, ag in enumerate(agents):
        df1 = cellline_df.loc[cellline_df['agent']==ag]
        targets.append(df1['target'].unique())
        concents = list(df1['concentration'].unique())
        concents_list = concents.copy()
        for j, con in enumerate(concents_list):
            df_temp = df1.loc[df1["concentration"] == con]
            if np.all(np.isnan(df_temp["G1"])):
                concents.pop(j)
        
        concentrations.append([1e-10] + concents)

        # for those agents that have less than 10 concentrations, set the last ln columns as nan
        if len(concents) < 10:
            ln = len(concents)
            tensor[:, :, :, ln+1:, i] = np.nan

        for j, cons in enumerate(concents[len(concents)-9:]):
            df2 = df1.loc[df1['concentration'] == cons]
            # for those that have less than 4 replicates, set the last l columns as nan
            l = len(df2)
            tensor[:, :, l:, j+1, i] = np.nan
            tensor[:, :, l:, 0, i] = np.nan

            # treatment
            # time = 72 hours
            tensor[0, 1, :l, j+1, i] = np.array(df2['G1']) * np.array(df2['cell_count'])
            tensor[1, 1, :l, j+1, i] = np.array(df2['S']) * np.array(df2['cell_count'])
            tensor[2, 1, :l, j+1, i] = (np.array(df2['G2']) + np.array(df2['M'])) * np.array(df2['cell_count'])
            tensor[3, 1, :l, j+1, i] = np.array(df2['dead_count']) + (np.array(df2['subG1']) + np.array(df2['beyondG2'])) * np.array(df2['cell_count'])

        # control
        ## time = 0
        cont = ctr.loc[ctr['timepoint'] == 'time0_ctrl']
        tensor[0, 0, :l, :, i] = np.array(cont.loc[cont['cell_line'] == cellline_name]['G1']) * np.array(df1['cell_count__time0'].unique())[0]
        tensor[1, 0, :l, :, i] = np.array(cont.loc[cont['cell_line'] == cellline_name]['S']) * np.array(df1['cell_count__time0'].unique())[0]
        tensor[2, 0, :l, :, i] = (np.array(cont.loc[cont['cell_line'] == cellline_name]['G2']) + np.array(cont.loc[cont['cell_line'] == cellline_name]['M'])) * np.array(df1['cell_count__time0'].unique())[0]
        tensor[3, 0, :l, :, i] = np.array(df1['dead_count__time0'])[0] + np.array(cont.loc[cont['cell_line'] == cellline_name]['subG1']) * np.array(df1['cell_count__time0'].unique())[0]

        ## time = 72 hours
        cont_72 = ctr.loc[ctr['timepoint'] == '72']
        tensor[0, 1, :l, 0, i] = np.array(cont_72.loc[cont_72['cell_line'] == cellline_name]['G1']) * np.array(df1['cell_count__ctrl'].unique())[0]
        tensor[1, 1, :l, 0, i] = np.array(cont_72.loc[cont_72['cell_line'] == cellline_name]['S']) * np.array(df1['cell_count__ctrl'].unique())[0]
        tensor[2, 1, :l, 0, i] = (np.array(cont_72.loc[cont_72['cell_line'] == cellline_name]['G2']) + np.array(cont_72.loc[cont_72['cell_line'] == cellline_name]['M'])) * np.array(df1['cell_count__ctrl'].unique())[0]
        tensor[3, 1, :l, 0, i] = np.array(df1['dead_count__ctrl'])[0] + np.array(cont_72.loc[cont_72['cell_line'] == cellline_name]['subG1']) * np.array(df1['cell_count__ctrl'].unique())[0]
    
    return tensor, agents, concentrations, targets

def hms_tensor():
    df_dict = separate_cellLines()
    del df_dict['BT474']

    slices = list()
    for line in df_dict:
        output = import_cellLine(line, df_dict[line])
        dat = output[0].reshape((8,4,280))
        # dat = output[0].reshape((8,4,10,28))
        # dat = dat.swapaxes(0,3).swapaxes(1,2)
        # dat = dat.reshape(280,4,8)
        sheet = np.nanmean(dat, axis=1)
        slices.append(sheet)
    
    return np.stack(slices,2)
    