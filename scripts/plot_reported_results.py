"""Plot rounded published aggregate values; no training or statistical recomputation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'reported_figures'
OUT.mkdir(parents=True,exist_ok=True)
frame=pd.read_csv(ROOT/'results/reported/fold_summary.csv')
for metric in ['macro_f1','macro_auc']:
    fig,ax=plt.subplots(figsize=(7,4.5))
    for cohort,offset,caption in [('full',-.13,'Full cohort (N=347)'),('age60',.13,'Age ≥60 (N=166)')]:
        data=frame[frame.cohort==cohort]
        ax.errorbar(data[f'{metric}_mean'],np.arange(7)+offset,xerr=data[f'{metric}_sd'],fmt='o',capsize=3,label=caption)
    labels=[f'{a} / {b}' for a,b in zip(data.init,data.fine_tune)]
    ax.set_yticks(np.arange(7),labels);ax.invert_yaxis()
    ax.set_xlabel('Macro-F1' if metric=='macro_f1' else 'Macro-AUC')
    ax.set_title('Reported five-fold mean ± SD (rounded source values)',fontsize=10)
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/f'{metric}.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/f'{metric}.pdf',bbox_inches='tight');plt.close(fig)
print('Saved reported-value plots to',OUT)
