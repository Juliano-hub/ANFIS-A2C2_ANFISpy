import sys
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))

diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))

if diretorio_pai not in sys.path:
    sys.path.append(diretorio_pai)

import traceback
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ANFISpy import ANFIS
from ANFISpy import aggregators
from ANFISpy import layer5
import time
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import gc
import ast

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from DatasetCases import getExecutionCase, getExecutionCaseEnergy
import csv
from imblearn.over_sampling import SMOTE
from collections import Counter

import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

def load_data(dataset_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    (
        file_name,
        path_relativo,
        selected_features,
        label_index,
    ) = getExecutionCase(dataset_id)

    full_path = os.path.normpath(os.path.join(script_dir, path_relativo))

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {full_path}")

    df = pd.read_csv(full_path, engine="python")
    
    # REMOVE DUPLICATAS
    df = df.drop_duplicates()

    # Conversão de nomes de colunas (strings) para índices
    if len(selected_features) > 0 and isinstance(selected_features[0], str):
        selected_features = [df.columns.get_loc(col) for col in selected_features]

    # Faz o mesmo para o label_index se ele tiver vindo como uma lista de string ou string única
    if isinstance(label_index, list) and isinstance(label_index[0], str):
        label_index = df.columns.get_loc(label_index[0])
    elif isinstance(label_index, str):
        label_index = df.columns.get_loc(label_index)

    df = df.dropna()

    feature_names = df.columns[selected_features]
    x = df.iloc[:, selected_features].values
    y_raw = df.iloc[:, label_index].values

    # Remover duplicatas (baseado em todas as colunas de features)
    #x, unique_idx = np.unique(x, axis=0, return_index=True)
    #y_raw = y_raw[unique_idx]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.ravel())

    return x, y, label_encoder, feature_names, file_name


def get_folds(x, y, kf):
    fold_splits = []

    for train_idx, test_idx in kf.split(x, y):
        fold_splits.append((train_idx, test_idx))

    return fold_splits

def instanceANFIS(and_op, deffuz_op, mf_names, mf_type, mf_number,label_encoder, feature_names):
    n_vars = len(x[0])

    variables = {
    'inputs': {
        'n_sets': n_vars * [mf_number],
        'uod': n_vars * [(0, 1)],
        'var_names': feature_names,
        'mf_names': n_vars * mf_names,
        },
    'output': {
        'var_names': label_encoder.classes_.tolist(),
        'n_classes': len(label_encoder.classes_),
        },
    }

    anfis = ANFIS(
        variables=variables, 
        mf_shape= mf_type,
        and_operator=and_op,
        layer_5_aggregator=deffuz_op,
    )
    return anfis

def pipelineANFIS(x, y, ANFIS_hyperparameters, label_encoder, feature_names):
    np.random.seed(ANFIS_hyperparameters["SEED"])
    torch.manual_seed(ANFIS_hyperparameters["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(ANFIS_hyperparameters["SEED"])
        torch.cuda.manual_seed_all(ANFIS_hyperparameters["SEED"])

    torch.backends.cudnn.benchmark = True

    random.seed(ANFIS_hyperparameters["SEED"])

    kf = StratifiedKFold(n_splits=ANFIS_hyperparameters["number_of_folds"], shuffle=True, random_state=ANFIS_hyperparameters["SEED"])
    
    fold_splits = get_folds(x, y, kf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_folds = []
    precision_folds = []
    recall_folds = []
    f1_folds = []

    inicio = time.time()
    for i, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"\nFold {i}")
        anfis = instanceANFIS(ANFIS_hyperparameters["and_op"], ANFIS_hyperparameters["deffuz_op"], ANFIS_hyperparameters["mf_names"], ANFIS_hyperparameters["mf_type"], ANFIS_hyperparameters["mf_number"], label_encoder, feature_names)
        anfis = anfis.to(device)

        #if i == 1:
        #    for name, param in anfis.named_parameters():
        #        print(f"Aqui: {name} | Tamanho: {param.shape}")
        #        print(param.data)
        #        print("-" * 30)
                
                #Camada: memberships.0.mu | Tamanho: torch.Size([3])
                #tensor([0.0000, 0.5000, 1.0000], device='cuda:0')

                #Camada: memberships.0.sigma | Tamanho: torch.Size([3])
                #tensor([0.2500, 0.2500, 0.2500], device='cuda:0')

                #Camada: memberships.1.mu | Tamanho: torch.Size([3])
                #tensor([0.0000, 0.5000, 1.0000], device='cuda:0')

                #Camada: memberships.1.sigma | Tamanho: torch.Size([3])
                #tensor([0.2500, 0.2500, 0.2500], device='cuda:0')

                #Camada: memberships.2.mu | Tamanho: torch.Size([3])
                #tensor([0.0000, 0.5000, 1.0000], device='cuda:0')

                #Camada: memberships.2.sigma | Tamanho: torch.Size([3])
                #tensor([0.2500, 0.2500, 0.2500], device='cuda:0')

                #Camada: memberships.3.mu | Tamanho: torch.Size([3])
                #tensor([0.0000, 0.5000, 1.0000], device='cuda:0')

                #Camada: memberships.3.sigma | Tamanho: torch.Size([3])
                #tensor([0.2500, 0.2500, 0.2500], device='cuda:0')


                                            # ([Num(Regras)* Num(classes), Num(atributos)]) = 81*3 = 243
                                            # largutapetala * a + larguracepala*b + alturapetala * c + alturacepala * d
                #consequents.linear.weight | Tamanho: torch.Size([243, 4])

                                            # Num(Regras)* Num(classes) = 243
                                            # um valor de bias para cada regra de cada classe
                #consequents.linear.bias | Tamanho: torch.Size([243])

        x_train = x[train_idx]
        y_train = y[train_idx]

        x_test = x[test_idx]
        y_test = y[test_idx]

        # Normalização
        #scaler = MinMaxScaler()
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.transform(x_test)

        result = trainModel(anfis, x_train, y_train, x_test, y_test, ANFIS_hyperparameters["epochs"], ANFIS_hyperparameters["batch_size"], device, ANFIS_hyperparameters["SEED"], ANFIS_hyperparameters["optimizer_type"], ANFIS_hyperparameters["learning_rate"])
        if result is None:
            print(f"!!! [ABORTANDO EXPERIMENTO] NaN detectado no Fold {i}. Pulando combinação...")
            return None

        metrics, y_pred, y_pred_round, train_losses, timeFold = result
        fold_dict = {
            "fold": i,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "confusion_matrix": metrics["confusion_matrix"],
            "y_pred": y_pred,
            "y_pred_round": y_pred_round,
            "y_real": y_test,
            "train_losses": train_losses,
            "time": timeFold,
            "report": metrics["report"],
            #"scaler_min": scaler.data_min_.tolist(),
            #"scaler_max": scaler.data_max_.tolist(),
            "test_idx": test_idx
        }
        
        accuracy_folds.append(metrics["accuracy"])
        precision_folds.append(metrics["precision"])
        recall_folds.append(metrics["recall"])
        f1_folds.append(metrics["f1_score"])

        save_folds_results(ANFIS_hyperparameters, fold_dict, anfis.get_MF(), anfis.print_rules(), feature_names)

    print(f'Average Accuracy: {np.mean(accuracy_folds):.4f} ± {np.std(accuracy_folds):.4f}')
    print(f'Average Precision: {np.mean(precision_folds):.4f} ± {np.std(precision_folds):.4f}')
    print(f'Average Recall: {np.mean(recall_folds):.4f} ± {np.std(recall_folds):.4f}')
    print(f'Average F1-score: {np.mean(f1_folds):.4f} ± {np.std(f1_folds):.4f}')
    display_time = str(datetime.timedelta(seconds=round((time.time() - inicio))))
    print(f"Execution time: {display_time} (H:MM:SS)")

    metrics = {
        "mean_accuracy": np.mean(accuracy_folds),
        "mean_precision": np.mean(precision_folds),
        "mean_recall": np.mean(recall_folds),
        "mean_F1-score": np.mean(f1_folds),
        "time": str(display_time)
    }

    return metrics

def SMOTESampling(X, Y, randomState, device):
    X_cpu = X.cpu().numpy()
    Y_cpu = Y.cpu().numpy()

    count = Counter(Y_cpu)
    print(f'Distribution before SMOTE: {count}')
    minorityClassCountEntry = min(count.values())
    smote = SMOTE(
        sampling_strategy='auto', # Define como o balanceamento será feito, auto ajusta a classe minoritária para ter o mesmo número de amostras da majoritária
        random_state=randomState, # Define a semente para reprodutibilidade
        k_neighbors= max(1, minorityClassCountEntry - 1)) # Número de vizinhos considerados para gerar os exemplos sintéticos                          

    X_resampled, y_resampled = smote.fit_resample(X_cpu, Y_cpu)
    
    print(f'Distribution after SMOTE: {Counter(y_resampled)}')

    X_resampled = torch.tensor(X_resampled, dtype=torch.float32).to(device)
    y_resampled = torch.tensor(y_resampled, dtype=torch.long).to(device)
    return X_resampled, y_resampled

def trainModel(anfis, x_train, y_train, x_test, y_test, epochs, batch_size, device, randomState, optimizer_type, learning_rate):
    start2 = time.time()
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    #x_val = torch.tensor(x_val, dtype=torch.float32)
    #y_val = torch.tensor(y_val, dtype=torch.long)

    #x_train, y_train = SMOTESampling(x_train, y_train, randomState, device)

    #x_train_max = x_train.max(dim=0, keepdim=True)[0]
    #x_train /= x_train_max
    #x_test /= x_train_max

    #print(len(x_train), len(y_train), len(x_test), len(y_test))
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(anfis.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(anfis.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type")

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        anfis.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = anfis(x_batch)
            # PROTEÇÃO PARA LOTE UNITÁRIO (Caso 31 pós drop_duplicates)
            # Se sobrou apenas 1 amostra no lote, ajusta a dimensão para a CrossEntropyLoss
            if y_batch.size(0) == 1 and len(y_pred.shape) == 1:
                y_pred = y_pred.unsqueeze(0)  # Transforma o vetor [3] em matriz [1, 3]
            loss = criterion(y_pred, y_batch)
            if torch.isnan(loss):
                print("Loss explodiu (NaN). Interrompendo este fold...")
                return None # Ou um dicionário de métricas zeradas
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        train_losses.append(train_loss)

        anfis.eval()
        #val_loss = 0.0
        
        #with torch.no_grad():
        #    for x_batch, y_batch in val_loader:
        #        y_pred = anfis(x_batch)
        #        loss = criterion(y_pred, y_batch)
        #        val_loss += loss.item()

        #val_loss /= len(val_loader)
        #val_losses.append(val_loss)
        if epoch % 10 == 9:
            print(f'Epoch {epoch+1}/{epochs}: train_loss: {train_loss:.4f}')

    metrics, y_pred, y_pred_round = evaluateModel(anfis, x_test, y_test, device)
    timeFold = str(datetime.timedelta(seconds=round((time.time() - start2))))
    return metrics, y_pred, y_pred_round, train_losses, timeFold

def save_folds_results(ANFIS_hyperparameters, fold_dict, mfFold, rulesFold, feature_names):
    base_folder = os.path.join("results", str(ANFIS_hyperparameters["folder_name"]))
    detail_folder = os.path.join(base_folder, f"{ANFIS_hyperparameters['dataset_id']}",f"{ANFIS_hyperparameters['and_op_name']}")
    os.makedirs(detail_folder, exist_ok=True)

    # Converte o tensor de scores [30, 3] em um array numpy
    y_pred_np = fold_dict["y_pred"].detach().cpu().numpy()

    # Cria o DataFrame base com as colunas de score
    df_preds = pd.DataFrame(
        y_pred_np, 
        columns=[f'score_class_{i}' for i in range(y_pred_np.shape[1])]
    )

    # Adiciona a classe prevista (y_pred_round)
    df_preds['y_pred_round'] = fold_dict["y_pred_round"].detach().cpu().numpy()
    
    # Adiciona a classe real
    df_preds['y_real'] = fold_dict["y_real"]

    # Coluna dos atributos originais
    # feature_names contém a lista: ['SepalLengthCm', 'SepalWidthCm', ...]
    # x[test_idx] garante os valores reais daquela rodada
    for idx, col_name in enumerate(feature_names):
            df_preds[col_name] = x[fold_dict['test_idx']][:, idx]

    # 6. Salva o DataFrame completo com tudo lado a lado
    df_preds.to_csv(os.path.join(detail_folder, f"fold_{fold_dict['fold']}_classes_previstas.csv"), index=False)


    with open(os.path.join(detail_folder, f"fold_{fold_dict["fold"]}_metrics.txt"), "w") as f:
        f.write(f"Fold: {fold_dict["fold"]}\n")
        f.write(f"Accuracy : {fold_dict["accuracy"]:.4f}\n")
        f.write(f"Precision: {fold_dict["precision"]:.4f}\n")
        f.write(f"Recall   : {fold_dict["recall"]:.4f}\n")
        f.write(f"F1-score : {fold_dict["f1_score"]:.4f}\n")
        f.write(f"TimeFold : {fold_dict["time"]}s\n")
        f.write(f"Last Train Loss: {fold_dict["train_losses"][-1] if len(fold_dict["train_losses"]) > 0 else 'N/A'}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{fold_dict["confusion_matrix"]}\n")
        f.write(f"{fold_dict["report"]}\n")


    # Criando o arquivo exclusivo para o histórico de Loss (com aspas simples na chave 'fold')
    with open(os.path.join(detail_folder, f"fold_{fold_dict['fold']}_loss_history.csv"), "w") as f_loss:
        # Escreve o cabeçalho
        f_loss.write("epoch,train_loss\n")
        
        # Varre a lista de perdas por época
        for epoch, loss_val in enumerate(fold_dict["train_losses"]):
            # epoch + 1 faz o arquivo começar na Época 1 em vez de 0
            f_loss.write(f"{epoch + 1},{loss_val}\n")

    with open(os.path.join(detail_folder, f"fold_{fold_dict["fold"]}_mf.txt"), "w") as f:
        for input_name, data in mfFold.items():
            f.write(f"{input_name} ({data['type']}):\n")
            for param_name, values in data.items():
                if param_name != 'type':
                    f.write(f"  {param_name}: {values.tolist()}\n")
            f.write("-" * 20 + "\n")

    with open(os.path.join(detail_folder, f"fold_{fold_dict["fold"]}_rules.txt"), "w") as f:
        for rule in rulesFold:
            f.write(f"{rule}\n")

    #with open(os.path.join(detail_folder, f"fold_{fold_dict["fold"]}_structureInfo.txt"), "w") as f:
    #        f.write(f"scaler_min: {fold_dict['scaler_min']}\n")
    #        f.write(f"scaler_max: {fold_dict['scaler_max']}\n")

def update_matrix_csv(file_path, dataset_id, dataset_name, col_name, value):
    # 1. Carrega ou cria o DataFrame
    if os.path.exists(file_path):
        # Lê o CSV usando o dataset_id como índice
        df = pd.read_csv(file_path, index_col='dataset_id')
    else:
        # Cria um DF novo com as colunas necessárias
        df = pd.DataFrame(columns=['dataset_name', col_name])
        df.index.name = 'dataset_id'

    # 2. Insere/Atualiza os dados
    # Define o nome do dataset naquela linha do ID
    df.at[dataset_id, 'dataset_name'] = dataset_name
    # Define o valor na coluna especificada
    df.at[dataset_id, col_name] = value
    
    # 3. Ordena pelo ID (índice)
    df = df.sort_index()

    # 4. Salva o arquivo mantendo o index (dataset_id)
    df.to_csv(file_path, index=True)

def save_general_results(dataset_name, ANFIS_hyperparameters, metrics):
    base_folder = os.path.join("results", str(ANFIS_hyperparameters["folder_name"]))
    os.makedirs(base_folder, exist_ok=True)

    acc_matrix_path = os.path.join(base_folder, "geral_accuracy_matrix.csv")
    pre_matrix_path = os.path.join(base_folder, "geral_precision_matrix.csv")
    re_matrix_path = os.path.join(base_folder, "geral_recall_matrix.csv")
    f1_matrix_path = os.path.join(base_folder, "geral_F1-score_matrix.csv")
    time_matrix_path = os.path.join(base_folder, "geral_time_matrix.csv")
    
    # Pega o nome do operador AND para ser a coluna
    and_op = ANFIS_hyperparameters["and_op_name"]
    
    # Atualiza a matriz de Acurácia
    update_matrix_csv(acc_matrix_path, ANFIS_hyperparameters["dataset_id"], dataset_name, and_op, metrics["mean_accuracy"])

    # Atualiza a matriz de Precisão
    update_matrix_csv(pre_matrix_path, ANFIS_hyperparameters["dataset_id"], dataset_name, and_op, metrics["mean_precision"])
    
    # Atualiza a matriz de Recall
    update_matrix_csv(re_matrix_path, ANFIS_hyperparameters["dataset_id"], dataset_name, and_op, metrics["mean_recall"])

    # Atualiza a matriz de Recall
    update_matrix_csv(f1_matrix_path, ANFIS_hyperparameters["dataset_id"], dataset_name, and_op, metrics["mean_F1-score"])

    # Atualiza a matriz de Tempo
    update_matrix_csv(time_matrix_path, ANFIS_hyperparameters["dataset_id"], dataset_name, and_op, metrics["time"])

def evaluateModel(anfis, x_test, y_test, device):
    anfis.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        y_pred = anfis(x_test)
        y_pred_round = torch.argmax(y_pred, dim=1)
        #[ 1.1128e+00, -6.8063e-01, -1.1006e+00]
        y_true = y_test.cpu().numpy()
        y_pred_np = y_pred_round.cpu().numpy()
        # [0]
        accuracy = accuracy_score(y_true, y_pred_np)
        precision = precision_score(y_true, y_pred_np, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_np, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred_np, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred_np)
        report = classification_report(y_true, y_pred_np, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "report": report
    }, y_pred, y_pred_round


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Use: python ANFIS-A2C2.py <dataset_id>")
        print("Ex: python ANFIS-A2C2.py 0")
        sys.exit(1)

    ANFIS_hyperparameters = {
    "dataset_id": int(sys.argv[1]),
    "and_op": None,
    "deffuz_op": None,
    "SEED": 10,
    "mf_names": [['Low', 'Medium', 'High']],
    "mf_type": 'gaussian',
    "mf_number": None,
    "epochs": 100,
    "number_of_folds": 5,
    "batch_size": 32,
    "and_op_name": None,
    "deffuz_op_name": None,
    "folder_name": None,
    "optimizer_type": 'adam',
    "learning_rate": 0.001
    }

    ANFIS_hyperparameters["mf_number"] = len(ANFIS_hyperparameters["mf_names"][0])


    aggregation_configs = [
        'ProdAND', 'ProdOR', 'LukasiewiczAND', 'LukasiewiczOR', 'MinAND', 'MaxOR',
        'FrankAND', 'FrankOR', 'DrasticAND', 'DrasticOR', 'YagerAND', 'YagerOR', 'SugenoWeberAND', 'SugenoWeberOR',
        'HamacherAND', 'HamacherOR', 'DombiAND', 'DombiOR', 'SchweizerSklarAND',
        'SchweizerSklarOR', 'AczelAlsinaAND', 'AczelAlsinaOR', 'NilpotentMin',
        'ArithmeticMean', 'GeometricMean', 'HarmonicMean', 'PowerMean', 'MedianAggregation',
        'Prospector', 'Example9Aggregation', 'Example10Aggregation',
        'Example11Aggregation', 'Example12Aggregation', 'Overlap', 'Grouping'
    ]

    # 'ProdAND', 'ProdOR', 'LukasiewiczAND', 'LukasiewiczOR', 'MinAND', 'MaxOR',
    #    'FrankAND', 'FrankOR', 'DrasticAND', 'DrasticOR', 'YagerAND', 'YagerOR', 'SugenoWeberAND', 'SugenoWeberOR',
    #    'HamacherAND', 'HamacherOR', 'DombiAND', 'DombiOR', 'SchweizerSklarAND',
    #    'SchweizerSklarOR', 'AczelAlsinaAND', 'AczelAlsinaOR', 'NilpotentMin',
    #    'ArithmeticMean', 'GeometricMean', 'HarmonicMean', 'PowerMean', 'MedianAggregation',
    #   'Prospector', 'Example9Aggregation', 'Example10Aggregation',
    #    'Example11Aggregation', 'Example12Aggregation', 'Overlap', 'Grouping'

    deffuz_configs = {
        'weighted_sum': lambda: layer5.WeightedSumANFISLayer5(),
    }

    #escala_padrao = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 150.0, 200.0]
    escala_padrao = [0.10, 0.50, 0.90, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    params_config = {
        'YagerAND': escala_padrao,
        'YagerOR':  escala_padrao,
        'SugenoWeberAND': escala_padrao,
        'SugenoWeberOR':  escala_padrao, 
        'SchweizerSklarAND': escala_padrao,
        'SchweizerSklarOR':  escala_padrao,
        'HamacherAND': escala_padrao,
        'HamacherOR':  escala_padrao,
        'DombiAND': escala_padrao,
        'DombiOR':  escala_padrao,
        'FrankAND': escala_padrao,
        'FrankOR':  escala_padrao,
        'AczelAlsinaAND': escala_padrao,
        'AczelAlsinaOR':  escala_padrao,
        'PowerMean': escala_padrao,
        'Example9Aggregation': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    x, y, label_encoder, feature_names, file_name = load_data(ANFIS_hyperparameters["dataset_id"])

    for def_name, def_factory in deffuz_configs.items():
        
        # Loop Intermediário: Operadores de Agregação
        for op_name in aggregation_configs:
            test_params = params_config.get(op_name, [None])
            
            for p in test_params:
                try:
                    # Instanciação dinâmica do agregador
                    agg_class = getattr(aggregators, op_name)
                    if p is not None:
                        current_agg = agg_class(p=p)
                        current_op_label = f"{op_name}_p{p:.2f}"
                    else:
                        current_agg = agg_class()
                        current_op_label = op_name
                    
                    # Criamos uma instância limpa da defuzzificação para cada teste
                    current_deffuz = def_factory()

                    print(f"\n" + "="*50)
                    print(f"Executando: Dataset={file_name} | AND={current_op_label} | Defuzz={def_name}")
                    print(f"="*50)
                    
                    # Atualiza hiperparâmetros
                    ANFIS_hyperparameters["and_op"] = current_agg
                    ANFIS_hyperparameters["and_op_name"] = current_op_label
                    ANFIS_hyperparameters["deffuz_op"] = current_deffuz
                    ANFIS_hyperparameters["deffuz_op_name"] = def_name

                    folder_name = (
                        f"def_{ANFIS_hyperparameters['deffuz_op_name']}_"    
                        f"mf_{ANFIS_hyperparameters['mf_type']}{ANFIS_hyperparameters['mf_number']}_"    
                        f"ep{ANFIS_hyperparameters['epochs']}_"               
                        f"f{ANFIS_hyperparameters['number_of_folds']}_"       
                        f"b{ANFIS_hyperparameters['batch_size']}_"           
                        f"s{ANFIS_hyperparameters['SEED']}"
                    )
                    ANFIS_hyperparameters["folder_name"] = folder_name

                    # Roda o treinamento e validação
                    metrics = pipelineANFIS(x, y, ANFIS_hyperparameters, label_encoder, feature_names)
                    
                    if metrics is not None:
                        save_general_results(file_name, ANFIS_hyperparameters, metrics)
                    else:
                        print(f"Experimento: Dataset={file_name} | AND={current_op_label} | Defuzz={def_name} descartado por instabilidade numérica.")
                        metrics_error = {
                            "mean_accuracy": np.nan,
                            "mean_precision": np.nan,
                            "mean_recall": np.nan,
                            "mean_F1-score": np.nan,
                            "time": "ERRO"
                        }
                        save_general_results(file_name, ANFIS_hyperparameters, metrics_error)
                        continue

                    # Salva os resultados (usando a lógica de pastas que discutimos)
                    save_general_results(file_name, ANFIS_hyperparameters, metrics)

                except Exception as e:
                    print(f"\n[ERRO] Falha no experimento {op_name} + {def_name}: {str(e)}")
                    traceback.print_exc()
                    continue 
                
                finally:
                    # Limpeza de memória
                    torch.cuda.empty_cache()
                    gc.collect()