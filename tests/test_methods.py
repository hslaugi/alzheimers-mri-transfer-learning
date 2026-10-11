"""Data-free checks; no MRI training, weight download, or GPU is needed."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,roc_auc_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1]
NB=json.loads((ROOT/'notebooks/Alzheimers_MRI_Transfer_Learning.ipynb').read_text())
SOURCE='\n'.join(c['source'] for c in NB['cells'] if c['cell_type']=='code')
NAMES={'make_subject_splits','calculate_metric','align_oof_predictions','paired_stratified_bootstrap'}
TREE=ast.parse(SOURCE)
SELECTED=ast.Module(body=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in NAMES],type_ignores=[])
NS=dict(np=np,pd=pd,StratifiedGroupKFold=StratifiedGroupKFold,f1_score=f1_score,roc_auc_score=roc_auc_score,
        balanced_accuracy_score=balanced_accuracy_score,SEED=42,N_FOLDS=5,NUM_CLASSES=3)
exec(compile(SELECTED,'selected_methods','exec'),NS)

class Methods(unittest.TestCase):
    def test_notebook_code_compiles(self):
        for i,c in enumerate(NB['cells']):
            if c['cell_type']=='code':
                compile(c['source'],f'cell_{i}','exec')
                self.assertEqual(c['outputs'],[])
                self.assertIsNone(c['execution_count'])
        self.assertEqual(ast.dump(ast.parse(SOURCE)),ast.dump(ast.parse((ROOT/'scripts/experiment.py').read_text())))

    def test_both_cohorts_disjoint_complete(self):
        for counts in [(266,58,23),(85,58,23)]:
            y=np.repeat(np.arange(3),counts)
            frame=pd.DataFrame({'subject_id':[f'OAS1_{i:04}' for i in range(len(y))],'label':y})
            splits=NS['make_subject_splits'](frame)
            held=[]
            for f in splits:
                tr,va,te=map(set,[f['train'],f['val'],f['test']])
                self.assertFalse(tr&va or tr&te or va&te)
                self.assertEqual(tr|va|te,set(frame.subject_id));held+=f['test']
            self.assertEqual(len(held),len(set(held)))
            self.assertEqual(set(held),set(frame.subject_id))

    def test_bootstrap_pairing_and_identical_model(self):
        y=np.tile(np.arange(3),10)
        prob=np.full((30,3),.1);prob[np.arange(30),y]=.8
        a=pd.DataFrame({'subject_id':[f's{i:03}' for i in range(30)],'true_label':y,'pred_label':y,'fold':1,
                        **{f'prob_class_{i}':prob[:,i] for i in range(3)}})
        for metric in ['macro_f1','macro_auc','balanced_accuracy']:
            result=NS['paired_stratified_bootstrap'](a,a.sample(frac=1,random_state=1),metric,n_boot=50)
            self.assertEqual(result['difference_A_minus_B'],0)
            self.assertEqual(result['CI_low'],0);self.assertEqual(result['CI_high'],0)
        bad=a.copy();bad.loc[0,'true_label']=1
        with self.assertRaises(AssertionError):NS['align_oof_predictions'](a,bad)
        with self.assertRaises(AssertionError):NS['align_oof_predictions'](pd.concat([a,a.iloc[:1]]),a)

    def test_reported_summary_shape(self):
        f=pd.read_csv(ROOT/'results/reported/fold_summary.csv')
        self.assertEqual(f.groupby('cohort').size().to_dict(),{'age60':7,'full':7})
        p=pd.read_csv(ROOT/'results/reported/paired_macro_f1.csv')
        self.assertEqual(len(p),6)
        self.assertTrue(((p.ci_low<0)&(p.ci_high>0)).all())

if __name__=='__main__':unittest.main()
