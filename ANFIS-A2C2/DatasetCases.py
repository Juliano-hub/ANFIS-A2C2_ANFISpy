
from matplotlib.pylab import conj

def getExecutionCase(ExecutionCase):
    File_Name = ''
    BasePath = 'Dataset/ComTratamento/'
    ColumnsX = []
    if ExecutionCase == 0:
        BasePath = 'Dataset'
        File_Name = 'IrisDataset'
        ColumnsX = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        ColumnsY = ["Species"]
    elif ExecutionCase == 1:
        File_Name = 'iqr_mc_CPU_BW_Storage'
        ColumnsX = ['CPU', 'BW', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 2:
        File_Name = 'iqr_mc_CPU_Energy_MIPS'
        ColumnsX = ['CPU', 'Energy', 'MIPS']
        ColumnsY = ['Class'] 
    elif ExecutionCase == 3:
        File_Name = 'iqr_mc_CPU_BW_MIPS'
        ColumnsX = ['CPU', 'BW', 'MIPS']
        ColumnsY = ['Class']  
    elif ExecutionCase == 4:
        File_Name = 'iqr_mc_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- iqr_mmt (Casos 5 ao 8) -----------------
    elif ExecutionCase == 5:
        File_Name = 'iqr_mmt_CPU_BW'
        ColumnsX = ['CPU', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 6:
        File_Name = 'iqr_mmt_CPU_Energy_Mem_Storage'
        ColumnsX = ['CPU', 'Energy', 'Mem', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 7:
        File_Name = 'iqr_mmt_CPU_BW_MIPS_Mem'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 8:
        File_Name = 'iqr_mmt_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- iqr_mu (Casos 9 e 10) -----------------
    elif ExecutionCase == 9:
        File_Name = 'iqr_mu_MIPS_Energy_CPU'
        ColumnsX = ['MIPS', 'Energy', 'CPU']
        ColumnsY = ['Class']
    elif ExecutionCase == 10:
        File_Name = 'iqr_mu_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- iqr_rs (Casos 11 ao 14) -----------------
    elif ExecutionCase == 11:
        File_Name = 'iqr_rs_CPU_MIPS_BW'
        ColumnsX = ['CPU', 'MIPS', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 12:
        File_Name = 'iqr_rs_CPU_Energy_MIPS_BW_Mem'
        ColumnsX = ['CPU', 'Energy', 'MIPS', 'BW', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 13:
        File_Name = 'iqr_rs_CPU_Storage_BW_Energy'
        ColumnsX = ['CPU', 'Storage', 'BW', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 14:
        File_Name = 'iqr_rs_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- lr_mc (Casos 15 ao 19) -----------------
    elif ExecutionCase == 15:
        File_Name = 'lr_mc_CPU_Storage_Mem_BW_MIPS_Energy'
        ColumnsX = ['CPU', 'Storage', 'Mem', 'BW', 'MIPS', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 16:
        File_Name = 'lr_mc_CPU_BW_Storage_MIPS_Mem'
        ColumnsX = ['CPU', 'BW', 'Storage', 'MIPS', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 17:
        File_Name = 'lr_mc_CPU_Energy_Storage_MIPS'
        ColumnsX = ['CPU', 'Energy', 'Storage', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 18:
        File_Name = 'lr_mc_CPU_BW_MIPS_Mem'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 19:
        File_Name = 'lr_mc_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 20:
        File_Name = 'lr_mmt_CPU_MIPS_Energy'
        ColumnsX = ['CPU', 'MIPS', 'Energy']
        ColumnsY = ['Class']
# ----------------- lr_mmt (Casos 21 ao 24) -----------------
    elif ExecutionCase == 21:
        File_Name = 'lr_mmt_CPU_BW_Energy'
        ColumnsX = ['CPU', 'BW', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 22:
        File_Name = 'lr_mmt_CPU_Energy_Storage_Mem'
        ColumnsX = ['CPU', 'Energy', 'Storage', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 23:
        File_Name = 'lr_mmt_CPU_BW_MIPS_Mem'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 24:
        File_Name = 'lr_mmt_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- lr_mu (Casos 25 ao 29) -----------------
    elif ExecutionCase == 25:
        File_Name = 'lr_mu_CPU_MIPS_Mem_Storage_Energy_BW'
        ColumnsX = ['CPU', 'MIPS', 'Mem', 'Storage', 'Energy', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 26:
        File_Name = 'lr_mu_CPU_BW_MIPS'
        ColumnsX = ['CPU', 'BW', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 27:
        File_Name = 'lr_mu_MIPS_Energy_CPU_Mem_BW'
        ColumnsX = ['MIPS', 'Energy', 'CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 28:
        File_Name = 'lr_mu_CPU_Mem_BW_Storage_Energy_MIPS'
        ColumnsX = ['CPU', 'Mem', 'BW', 'Storage', 'Energy', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 29:
        File_Name = 'lr_mu_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- lr_rs (Casos 30 ao 34) -----------------
    elif ExecutionCase == 30:
        File_Name = 'lr_rs_CPU_Storage_BW_Mem_MIPS_Energy'
        ColumnsX = ['CPU', 'Storage', 'BW', 'Mem', 'MIPS', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 31:
        File_Name = 'lr_rs_CPU_BW_Energy'
        ColumnsX = ['CPU', 'BW', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 32:
        File_Name = 'lr_rs_Mem_Energy_CPU_MIPS'
        ColumnsX = ['Mem', 'Energy', 'CPU', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 33:
        File_Name = 'lr_rs_CPU_Mem_BW_Storage_Energy_MIPS'
        ColumnsX = ['CPU', 'Mem', 'BW', 'Storage', 'Energy', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 34:
        File_Name = 'lr_rs_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- lrr_mc (Casos 35 ao 39) -----------------
    elif ExecutionCase == 35:
        File_Name = 'lrr_mc_CPU_Storage_Mem_BW_MIPS_Energy'
        ColumnsX = ['CPU', 'Storage', 'Mem', 'BW', 'MIPS', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 36:
        File_Name = 'lrr_mc_CPU_BW_Storage_MIPS_Mem'
        ColumnsX = ['CPU', 'BW', 'Storage', 'MIPS', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 37:
        File_Name = 'lrr_mc_CPU_Energy_Storage_MIPS'
        ColumnsX = ['CPU', 'Energy', 'Storage', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 38:
        File_Name = 'lrr_mc_CPU_BW_MIPS_Mem'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem']
        ColumnsY = ['Class']  
    elif ExecutionCase == 39:
        File_Name = 'lrr_mc_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
# ----------------- lrr_mmt (Casos 41 ao 44) -----------------
    elif ExecutionCase == 40:
        File_Name = 'lrr_mmt_CPU_MIPS_Energy'
        ColumnsX =  ["CPU","MIPS","Energy"]
        ColumnsY = ['Class']
    elif ExecutionCase == 41:
        File_Name = 'lrr_mmt_CPU_BW_Energy'
        ColumnsX = ['CPU', 'BW', 'Energy']
        ColumnsY = ['Class']
    elif ExecutionCase == 42:
        File_Name = 'lrr_mmt_CPU_Energy_Storage_Mem'
        ColumnsX = ['CPU', 'Energy', 'Storage', 'Mem']
        ColumnsY = ['Class']
    elif ExecutionCase == 43:
        File_Name = 'lrr_mmt_CPU_BW_MIPS_Mem_Storage'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 44:
        File_Name = 'lrr_mmt_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- lrr_mu (Casos 45 ao 48) -----------------
    elif ExecutionCase == 45:
        File_Name = 'lrr_mu_CPU_BW_MIPS'
        ColumnsX = ['CPU', 'BW', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 46:
        File_Name = 'lrr_mu_MIPS_Energy_CPU_Mem_BW'
        ColumnsX = ['MIPS', 'Energy', 'CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 47:
        File_Name = 'lrr_mu_CPU_BW_MIPS_Mem_Storage'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 48:
        File_Name = 'lrr_mu_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- lrr_rs (Casos 49 ao 53) -----------------
    elif ExecutionCase == 49:
        File_Name = 'lrr_rs_CPU_Mem_BW_MIPS_Energy_Storage'
        ColumnsX = ['CPU', 'Mem', 'BW', 'MIPS', 'Energy', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 50:
        File_Name = 'lrr_rs_CPU_BW_Storage'
        ColumnsX = ['CPU', 'BW', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 51:
        File_Name = 'lrr_rs_CPU_Energy_Storage'
        ColumnsX = ['CPU', 'Energy', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 52:
        File_Name = 'lrr_rs_CPU_BW_MIPS_Mem_Storage'
        ColumnsX = ['CPU', 'BW', 'MIPS', 'Mem', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 53:
        File_Name = 'lrr_rs_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- mad_mc (Casos 54 e 55) -----------------
    elif ExecutionCase == 54:
        File_Name = 'mad_mc_CPU_Energy_MIPS'
        ColumnsX = ['CPU', 'Energy', 'MIPS']
        ColumnsY = ['Class']
    elif ExecutionCase == 55:
        File_Name = 'mad_mc_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- mad_mmt (Casos 56 e 57) -----------------
    elif ExecutionCase == 56:
        File_Name = 'mad_mmt_CPU_Energy_Mem_Storage'
        ColumnsX = ['CPU', 'Energy', 'Mem', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 57:
        File_Name = 'mad_mmt_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- mad_mu (Casos 58 e 59) -----------------
    elif ExecutionCase == 58:
        File_Name = 'mad_mu_CPU_Energy_Mem_BW'
        ColumnsX = ['CPU', 'Energy', 'Mem', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 59:
        File_Name = 'mad_mu_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- mad_rs (Casos 60 e 61) -----------------
    elif ExecutionCase == 60:
        File_Name = 'mad_rs_MIPS_Energy_CPU_Mem_Storage'
        ColumnsX = ['MIPS', 'Energy', 'CPU', 'Mem', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 61:
        File_Name = 'mad_rs_CPU_Mem_BW'
        ColumnsX = ['CPU', 'Mem', 'BW']
        ColumnsY = ['Class']
    # ----------------- Casos datasets especiais -----------------
    elif ExecutionCase == 62:
        BasePath = 'Dataset'
        File_Name = 'job_inf_df_Alibaba2026_SelectedFeatures_dropduplicatas_Amostra_38974'
        ColumnsX = ["cpu_request", "gpu_request", "worker_num", "duration"]
        ColumnsY = ["job_type"]
    elif ExecutionCase == 63:
        BasePath = 'Dataset'
        File_Name = 'Google2019ClusterSample_SelectedFeatures_dropduplicatas'
        #ColumnsX = ["resource_request_cpus","resource_request_memory","average_usage_cpus","average_usage_memory","maximum_usage_cpus","maximum_usage_memory","assigned_memory"]
        ColumnsX = ["average_usage_cpus","average_usage_memory","maximum_usage_cpus","maximum_usage_memory","assigned_memory"]
        ColumnsY = ["failed"]
    elif ExecutionCase == 64:
        BasePath = 'Dataset'
        File_Name = 'Azure2019 - trace_data_vmtable_vmtable_38974'
        ColumnsX = ["maxcpu","avgcpu","vmcorecountbucket","vmmemorybucket","lifetime"]
        ColumnsY = ["vmcategory"]
    # ----------------- ICEIS (Casos 65 ao 68) -----------------
    elif ExecutionCase == 65:
        BasePath = 'Dataset/ICEIS'
        File_Name = 'iqr_rs_CPU_Mem_BW_Storage'
        ColumnsX = ['CPU', 'Mem', 'BW', 'Storage']
        ColumnsY = ['Class']
    elif ExecutionCase == 66:
        BasePath = 'Dataset/ICEIS'
        File_Name = 'iqr_rs_CPU_Energy_Storage_BW'
        ColumnsX = ['CPU', 'Energy', 'Storage', 'BW']
        ColumnsY = ['Class']
    elif ExecutionCase == 67:
        BasePath = 'Dataset/ICEIS'
        File_Name = 'cloudsim_iqr_rs_cpu_mem_bw_storage'
        ColumnsX = ['CPU', 'Mem', 'BW', 'Storage'] # Mantido em minúsculas conforme o arquivo
        ColumnsY = ['Class']
    elif ExecutionCase == 68:
        BasePath = 'Dataset/ICEIS'
        File_Name = 'cloudsim_iqr_rs_cpu_energy_storage_bw'
        ColumnsX = ['CPU', 'Energy', 'Storage', 'BW'] # Mantido em minúsculas conforme o arquivo
        ColumnsY = ['Class']

    #mf, conj = getMFsAmount(FeaturesFromDataset)
    CSVFile = f'{BasePath}/{File_Name}.csv'
    #return File_Name, CSVFile, ColumnsX, ColumnsY, mf, conj
    return File_Name, CSVFile, ColumnsX, ColumnsY


def getExecutionCaseEnergy(ExecutionCase):
    File_Name = ''
    BasePath = 'Dataset/ComTratamento/'
    ColumnsX = []
    ColumnsY = []
    if ExecutionCase == 2:
        File_Name = 'iqr_mc_CPU_Energy_MIPS'
        ColumnsX =  ['CPU', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 6:
        File_Name = 'iqr_mmt_CPU_Energy_Mem_Storage'
        ColumnsX = ['CPU', 'Mem', 'Storage']
        ColumnsY = ['Energy']
    #-----------------iqr_mmt-----------------
    elif ExecutionCase == 9:
        File_Name = 'iqr_mu_MIPS_Energy_CPU'
        ColumnsX =  ['MIPS','CPU']
        ColumnsY = ['Energy']
    #-----------------iqr_mu-----------------
    elif ExecutionCase == 12:
        File_Name = 'iqr_rs_CPU_Energy_MIPS_BW_Mem'
        ColumnsX = ['CPU', 'MIPS', 'BW', 'Mem']
        ColumnsY = ['Energy']
    elif ExecutionCase == 13:
        File_Name = 'iqr_rs_CPU_Storage_BW_Energy'
        ColumnsX = ['CPU', 'Storage', 'BW']
        ColumnsY = ['Energy']
    #-----------------iqr_rs-----------------
    elif ExecutionCase == 15:
        File_Name = 'lr_mc_CPU_Storage_Mem_BW_MIPS_Energy'
        ColumnsX = ['CPU', 'Storage', 'Mem', 'BW', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 17:
        File_Name = 'lr_mc_CPU_Energy_Storage_MIPS'
        ColumnsX =  ['CPU', 'Storage', 'MIPS']
        ColumnsY = ['Energy']
    #-----------------lr_mc-----------------
    elif ExecutionCase == 20:
        File_Name = 'lr_mmt_CPU_MIPS_Energy'
        ColumnsX =  ['CPU', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 21:
        File_Name = 'lr_mmt_CPU_BW_Energy'
        ColumnsX =  ['CPU', 'BW']
        ColumnsY = ['Energy']
    elif ExecutionCase == 22:
        File_Name = 'lr_mmt_CPU_Energy_Storage_Mem'
        ColumnsX =  ['CPU', 'Storage', 'Mem']
        ColumnsY = ['Energy']
    #-----------------lr_mmt-----------------
    elif ExecutionCase == 25:
        File_Name = 'lr_mu_CPU_MIPS_Mem_Storage_Energy_BW'
        ColumnsX =  ['CPU', 'MIPS', 'Mem', 'Storage', 'BW']
        ColumnsY = ['Energy']
    elif ExecutionCase == 27:
        File_Name = 'lr_mu_MIPS_Energy_CPU_Mem_BW'
        ColumnsX =  ['MIPS', 'CPU', 'Mem', 'BW']
        ColumnsY = ['Energy']
    elif ExecutionCase == 28:
        File_Name = 'lr_mu_CPU_Mem_BW_Storage_Energy_MIPS'
        ColumnsX =  ['CPU', 'Mem', 'BW', 'Storage', 'MIPS']
        ColumnsY = ['Energy']
    #-----------------lr_mu-----------------
    elif ExecutionCase == 30:
        File_Name = 'lr_rs_CPU_Storage_BW_Mem_MIPS_Energy'
        ColumnsX =  ['CPU', 'Storage', 'BW', 'Mem', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 31:
        File_Name = 'lr_rs_CPU_BW_Energy'
        ColumnsX =  ['CPU', 'BW']
        ColumnsY = ['Energy']
    elif ExecutionCase == 32:
        File_Name = 'lr_rs_Mem_Energy_CPU_MIPS'
        ColumnsX =  ['Mem', 'CPU', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 33:
        File_Name = 'lr_rs_CPU_Mem_BW_Storage_Energy_MIPS'
        ColumnsX =  ['CPU', 'Mem', 'BW', 'Storage', 'MIPS']
        ColumnsY = ['Energy']
    #-----------------lr_rs-----------------
    elif ExecutionCase == 35:
        File_Name = 'lrr_mc_CPU_Storage_Mem_BW_MIPS_Energy'
        ColumnsX =  ['CPU', 'Storage', 'Mem', 'BW', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 37:
        File_Name = 'lrr_mc_CPU_Energy_Storage_MIPS'
        ColumnsX =  ['CPU', 'Storage', 'MIPS']
        ColumnsY = ['Energy']
    #-----------------lrr_mc-----------------
    elif ExecutionCase == 40:
        File_Name = 'lrr_mmt_CPU_MIPS_Energy'
        ColumnsX =  ['CPU', 'MIPS']
        ColumnsY = ['Energy']
    elif ExecutionCase == 41:
        File_Name = 'lrr_mmt_CPU_BW_Energy'
        ColumnsX =  ['CPU', 'BW']
        ColumnsY = ['Energy']
    elif ExecutionCase == 42:
        File_Name = 'lrr_mmt_CPU_Energy_Storage_Mem'
        ColumnsX =  ['CPU', 'Storage', 'Mem']
        ColumnsY = ['Energy']
    #-----------------lrr_mmt-----------------
    elif ExecutionCase == 46:
        File_Name = 'lrr_mu_MIPS_Energy_CPU_Mem_BW'
        ColumnsX =  ['MIPS', 'CPU', 'Mem', 'BW']
        ColumnsY = ['Energy']
    #-----------------lrr_mu-----------------
    elif ExecutionCase == 49:
        File_Name = 'lrr_rs_CPU_Mem_BW_MIPS_Energy_Storage'
        ColumnsX =  ['CPU', 'Mem', 'BW', 'MIPS', 'Energy']
        ColumnsY = ['Energy']
    elif ExecutionCase == 51:
        File_Name = 'lrr_rs_CPU_Energy_Storage'
        ColumnsX =  ['CPU', 'Storage']
        ColumnsY = ['Energy']
    #-----------------lrr_rs-----------------
    elif ExecutionCase == 54:
        File_Name = 'mad_mc_CPU_Energy_MIPS'
        ColumnsX =  ['CPU', 'MIPS']
        ColumnsY = ['Energy']
    #-----------------mad_mc-----------------
    elif ExecutionCase == 56:
        File_Name = 'mad_mmt_CPU_Energy_Mem_Storage'
        ColumnsX =  ['CPU', 'Mem', 'Storage']
        ColumnsY = ['Energy']
    #-----------------mad_mmt-----------------
    elif ExecutionCase == 58:
        File_Name = 'mad_mu_CPU_Energy_Mem_BW'
        ColumnsX =  ['CPU', 'Mem', 'BW']
        ColumnsY = ['Energy']
    #-----------------mad_rs-----------------
    elif ExecutionCase == 60:
        File_Name = 'mad_rs_MIPS_Energy_CPU_Mem_Storage'
        ColumnsX =  ['MIPS', 'CPU', 'Mem', 'Storage']
        ColumnsY = ['Energy']

    #FeaturesFromDataset = len(ColumnsX)

    #mf, conj = getMFsAmount(FeaturesFromDataset)

    CSVFile = f'{BasePath}/{File_Name}.csv'
    #return File_Name, CSVFile, ColumnsX, ColumnsY, mf, conj
    return File_Name, CSVFile, ColumnsX, ColumnsY

def getMFsAmount(FeaturesFromDataset):
    mf = []
    conj = []
    if FeaturesFromDataset == 2:
        mf = [
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]]
        ]
    if FeaturesFromDataset == 3:
        mf = [
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]]
        ]
    elif FeaturesFromDataset == 4:
        mf = [
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]]
        ]
        #mf = [
        #  [['gaussmf',{'mean':0.,'sigma':1.}],
        #   ['gaussmf',{'mean':-1.,'sigma':2.}],
        #   ['gaussmf',{'mean':-4.,'sigma':10.}]],
        #  [['gaussmf',{'mean':1.,'sigma':2.}],
        #   ['gaussmf',{'mean':2.,'sigma':3.}], 
        #   ['gaussmf',{'mean':-2.,'sigma':10.}]],
        #  [['gaussmf',{'mean':1.,'sigma':2.}],
        #   ['gaussmf',{'mean':2.,'sigma':3.}], 
        #   ['gaussmf',{'mean':-2.,'sigma':10.}]],
        #  [['gaussmf',{'mean':1.,'sigma':2.}],
        #   ['gaussmf',{'mean':2.,'sigma':3.}], 
        #   ['gaussmf',{'mean':-2.,'sigma':10.}]
        #   ]]
    elif FeaturesFromDataset == 5:
        mf = [
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]]
        ]
    elif FeaturesFromDataset == 6:
        mf = [
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]],
        [['gaussmf',{'mean':0.0,'sigma':0.15}],['gaussmf',{'mean':0.5,'sigma':0.15}],['gaussmf',{'mean':1.0,'sigma':0.15}]]
        ]
    
    conj = ['baixo', 'medio', 'alto']
    return mf, conj