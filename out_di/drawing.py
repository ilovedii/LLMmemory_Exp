import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('diversity.csv') 

def plot_comparison_from_csv(df, metric_column, title, ylabel, filename):
    mem0_data = df[df['group'].str.startswith('mem0_')].copy()
    none_data = df[df['group'].str.startswith('none_')].copy()
    
    mem0_data['base_group'] = mem0_data['group'].str.replace('mem0_', '')
    none_data['base_group'] = none_data['group'].str.replace('none_', '')
    
    common_groups = sorted(set(mem0_data['base_group']) & set(none_data['base_group']))
    
    values_with = []
    values_without = []
    
    for group in common_groups:
        mem0_row = mem0_data[mem0_data['base_group'] == group]
        none_row = none_data[none_data['base_group'] == group]
        
        if not mem0_row.empty and not none_row.empty:
            values_with.append(mem0_row[metric_column].iloc[0])
            values_without.append(none_row[metric_column].iloc[0])
        else:
            values_with.append(np.nan)
            values_without.append(np.nan)
    
    # 繪圖
    x = np.arange(len(common_groups))
    bar_width = 0.25
    
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - (bar_width/2 + 0.05), values_without, bar_width, 
            label='without mem0', color='#6aaac4')
    plt.bar(x + (bar_width/2 + 0.05), values_with, bar_width, 
            label='with mem0', color='#87ceeb', hatch='///', edgecolor='white')
    
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.6, 0.1))

    plt.xticks(x, common_groups, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'out_pic/{filename}')

plot_comparison_from_csv(df,'lex_jsd','Lexical Diversity (↑)','Lexical Entropy','lex.png')
# plot_comparison_from_csv(df, 'SDI', 'SDI Comparison', 'SDI Score', 'sdi_comparison.png')
# plot_comparison_from_csv(df, 'distinct2', 'Distinct-2 Comparison', 'Distinct-2 Score', 'distinct2_comparison.png')
# plot_comparison_from_csv(df, 'selfbleu', 'Self-BLEU Comparison', 'Self-BLEU Score', 'selfbleu_comparison.png')
# plot_comparison_from_csv(df, 'plot_mpd', 'Plot MPD Comparison', 'Plot MPD Score', 'plot_mpd_comparison.png')