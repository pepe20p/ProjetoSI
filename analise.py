import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def analyze_spoofing_results(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado.")
        return


    df['y_true'] = df['is_real'].astype(int)

    df['y_pred'] = df['output'].apply(lambda x: 1 if str(x).strip().lower() == 'real' else 0)

    programs = df['program'].unique()
    
    metrics_summary = []

    n_programs = len(programs)
    fig, axes = plt.subplots(1, n_programs, figsize=(6 * n_programs, 5))
    if n_programs == 1: axes = [axes] # Garantir que axes seja iterável se houver apenas 1 programa

    print(f"{'='*60}")
    print(f"{'RELATÓRIO DE PERFORMANCE POR ALGORITMO':^60}")
    print(f"{'='*60}\n")

    for idx, prog in enumerate(programs):
        subset = df[df['program'] == prog]
        
        y_true = subset['y_true']
        y_pred = subset['y_pred']

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics_summary.append({
            'Algoritmo': prog,
            'Acurácia': acc,
            'Precisão': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        print(f"--- Algoritmo: {prog.upper()} ---")
        print(f"Total de amostras: {len(subset)}")
        print(f"Acurácia: {acc:.2%}")
        print(f"Precisão (capacidade de não classificar Spoof como Real): {prec:.2%}")
        print(f"Recall (capacidade de detectar todos os Reais): {rec:.2%}")
        print(f"F1-Score: {f1:.2%}\n")

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Pred: Spoof', 'Pred: Real'],
                    yticklabels=['Real: Spoof', 'Real: Real'])
        axes[idx].set_title(f'Matriz de Confusão: {prog.upper()}')
        axes[idx].set_xlabel('Predito')
        axes[idx].set_ylabel('Verdadeiro')

    plt.tight_layout()
    plt.show()

    # 3. Gráfico Comparativo de Métricas
    metrics_df = pd.DataFrame(metrics_summary)
    
    metrics_melted = metrics_df.melt(id_vars="Algoritmo", var_name="Métrica", value_name="Valor")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x="Algoritmo", y="Valor", hue="Métrica", palette="viridis")
    plt.title("Comparativo de Performance entre Algoritmos")
    plt.ylim(0, 1.1) 
    plt.ylabel("Score (0-1)")
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_spoofing_results('results.csv')