import json
import matplotlib.pyplot as plt
import glob
import numpy as np

files = sorted(glob.glob('*.json'))

groups = {}
for f in files:
    name = f.split('/')[-1].replace('.json', '')
    
    if name.startswith('mem0_'):
        group = name.replace('mem0_', '')
        mem_type = 'with'
    elif name.startswith('none_'):
        group = name.replace('none_', '')
        mem_type = 'without'
    else:
        continue
    if group not in groups:
        groups[group] = {}
    groups[group][mem_type] = f

group_names = sorted(groups.keys())

jac_median_with = []
jac_median_without = []
jac_low_ratio_with = []
jac_low_ratio_without = []
main_persist_with = []
main_persist_without = []

nli_with = []
nli_without = []

perplexity_with = []
perplexity_without = []

coherence_with = []
coherence_without = []

for group in group_names:
    f = groups[group].get('without')
    if f:
        with open(f, 'r') as fp:
            data = json.load(fp)
            # nli_without.append(data['NLI']['nli_contradiction_rate']['value'])
            
            # perplexity_without.append(data['Perplexity']['ppl3_add1']['value'])

            jac_median_without.append(data['NER(Jaccard)']['jac_median']['value'])
            main_persist_without.append(data['NER(Jaccard)']['main_persist']['value'])
            jac_low_ratio_without.append(data['NER(Jaccard)']['jac_low_ratio']['value'])

            # coherence_without.append(data['Thematic coherence']['global_coherence']['value'])
    else:
        # nli_without.append(np.nan)

        # perplexity_without.append(np.nan)

        jac_median_without.append(np.nan)
        main_persist_without.append(np.nan)
        jac_low_ratio_without.append(np.nan)

        # coherence_without.append(np.nan)

    f = groups[group].get('with')
    if f:
        with open(f, 'r') as fp:
            data = json.load(fp)
            # nli_with.append(data['NLI']['nli_contradiction_rate']['value'])

            # perplexity_with.append(data['Perplexity']['ppl3_add1']['value'])

            jac_median_with.append(data['NER(Jaccard)']['jac_median']['value'])
            main_persist_with.append(data['NER(Jaccard)']['main_persist']['value'])
            jac_low_ratio_with.append(data['NER(Jaccard)']['jac_low_ratio']['value'])

            # coherence_with.append(data['Thematic coherence']['global_coherence']['value'])
    else:
        # nli_with.append(np.nan)

        # perplexity_with.append(np.nan)

        jac_median_with.append(np.nan)
        main_persist_with.append(np.nan)
        jac_low_ratio_with.append(np.nan)

        # coherence_with.append(np.nan)

x = np.arange(len(group_names))
bar_width = 0.08

plt.figure(figsize=(12, 6))
# plt.bar(x - (bar_width/2+ 0.05), nli_without, bar_width, label='without mem0', color='#6aaac4')
# plt.bar(x + (bar_width/2+ 0.05), nli_with, bar_width, label='with mem0', color='#87ceeb', hatch='///', edgecolor='white')

offsets = np.linspace(-bar_width*4, bar_width*4, 6)
plt.bar(x + offsets[0], jac_median_without, bar_width, label='jac_median (without mem0)', color='#6aaac4')
plt.bar(x + offsets[1], jac_median_with, bar_width, label='jac_median (with mem0)', color='#87ceeb', hatch='////', edgecolor='white')
plt.bar(x + offsets[4], main_persist_without, bar_width, label='main_persist (without mem0)', color="#FF8952")
plt.bar(x + offsets[5], main_persist_with, bar_width, label='main_persist (with mem0)', color="#F58D5C", hatch='////', edgecolor='white')
plt.bar(x + offsets[2], jac_low_ratio_without, bar_width, label='jac_low_ratio (without mem0)', color="#919090")
plt.bar(x + offsets[3], jac_low_ratio_with, bar_width, label='jac_low_ratio (with mem0)', color="#C2C2C2", hatch='////', edgecolor='white')



# plt.bar(x - (bar_width/2+ 0.05), perplexity_without, bar_width, label='without mem0', color='#6aaac4')
# plt.bar(x + (bar_width/2+ 0.05), perplexity_with, bar_width, label='with mem0', color='#87ceeb', hatch='///', edgecolor='white')
# plt.ylim(300, 360)
# plt.yticks(np.arange(300, 361, 10))

# plt.bar(x - (bar_width/2+ 0.05), coherence_without, bar_width, label='without mem0', color='#6aaac4')
# plt.bar(x + (bar_width/2+ 0.05), coherence_with, bar_width, label='with mem0', color='#87ceeb', hatch='///', edgecolor='white')
# plt.ylim(0.9, 1.0)
# plt.yticks(np.arange(0.9, 1.01, 0.02))


plt.xticks(x, group_names, rotation=45, ha='right')
plt.ylabel('Value')
# plt.title('NLI Contradiction Rate (↓ is better)')
plt.title('NER Jaccard Comparison')
# plt.title('Perplexity Comparison (↓ is better)')
# plt.title('Thematic Coherence Comparison (↑ is better)')
plt.legend()
plt.tight_layout()
plt.savefig('out_pic/ner.png')